from pyomo.environ import AbstractModel,Set,Param,Var,Objective,Constraint,SolverFactory
from pyomo.environ import NonNegativeReals, PositiveReals, inequality
from pyomo.environ import value


def create_model():

    model = AbstractModel()

    ## SETS
    model.T = Set(dimen=1) # Periods
    model.H = Set(dimen=1) # Houses
    model.D = Set(dimen=1) # Loads
    model.G = Set(dimen=1) # PVs
    model.B = Set(dimen=1) # Batteries


    ## PARAMETERS
    model.demand                        = Param(model.T, model.D)
    model.generation                    = Param(model.T, model.G)
    
    model.battery_min_soc               = Param(model.B, within=NonNegativeReals)
    model.battery_capacity              = Param(model.B, within=NonNegativeReals)
    model.battery_charge_max            = Param(model.B, within=NonNegativeReals)
    model.battery_discharge_max         = Param(model.B, within=NonNegativeReals)
    model.battery_efficiency_charge     = Param(model.B, within=NonNegativeReals)
    model.battery_efficiency_discharge  = Param(model.B, within=NonNegativeReals)
    model.battery_soc_ini               = Param(model.B, within=NonNegativeReals)
    model.battery_soc_fin               = Param(model.B, within=NonNegativeReals)
    
    model.marketmakerrate               = Param(model.T)
    model.feedintariff                  = Param(model.T)
    model.community_fee                 = Param(model.T)
    model.grid_fee                      = Param(model.T)
    
    model.dt                            = Param(within=PositiveReals, default=1.0)

    
    ## VARIABLES
    # Number of servings consumed of each food
    model.COST_ENERGY = Var(model.T)
    model.COST_GRID   = Var(model.T)
    model.PL1_BUY     = Var(model.T, model.H, within=NonNegativeReals)
    model.PL1_SELL    = Var(model.T, model.H, within=NonNegativeReals)
    model.PL2_BUY     = Var(model.T, within=NonNegativeReals)
    model.PL2_SELL    = Var(model.T, within=NonNegativeReals)    
    model.B_SOC       = Var(model.T, model.B, within=NonNegativeReals)
    model.B_IN        = Var(model.T, model.B, within=NonNegativeReals)
    model.B_OUT       = Var(model.T, model.B, within=NonNegativeReals)
    model.CO2         = Var(model.T)


    ## OBJECTIVE
    # Minimize cost
    def total_cost(model):
        return sum(model.COST_ENERGY[t] + model.COST_GRID[t] for t in model.T)
    model.total_cost = Objective(rule=total_cost)


    ## VARIABLE LIMITS
    def soc_limits(model, t, b):
        return inequality(model.battery_min_soc[b], model.B_SOC[t,b], model.battery_capacity[b])
    model.soc_limits = Constraint(model.T, model.B, rule=soc_limits)

    def charge_limits(model, t, b):
        return inequality(0.0, model.B_IN[t,b], model.battery_charge_max[b])
    model.charge_limits = Constraint(model.T, model.B, rule=charge_limits)

    def discharge_limits(model, t, b):
        return inequality(0.0, model.B_OUT[t,b], model.battery_discharge_max[b])
    model.discharge_limits = Constraint(model.T, model.B, rule=discharge_limits)    


    ## CONSTRAINTS
    # Total energy cost per period  
    def energy_cost(model, t):
        return model.COST_ENERGY[t] == model.marketmakerrate[t]*model.PL2_BUY[t]*model.dt - model.feedintariff[t]*model.PL2_SELL[t]*model.dt
    model.energy_cost = Constraint(model.T, rule=energy_cost)    
    
    # Total grid cost per period
    def grid_cost(model, t):
        return model.COST_GRID[t] ==  sum( model.community_fee[t]*model.PL1_BUY[t,h]*model.dt for h in model.H ) \
                                + model.grid_fee[t]*model.PL2_BUY[t]*model.dt \
                                + (model.grid_fee[t]+model.community_fee[t])*model.PL2_SELL[t]*model.dt
    model.grid_cost = Constraint(model.T, rule=grid_cost)     
        
    
    # Community energy balance
    def energy_balance_grid(model, t):
        return model.PL2_SELL[t] - model.PL2_BUY[t] == sum( model.PL1_SELL[t,h] - model.PL1_BUY[t,h] for h in model.H )
    model.energy_balance_grid = Constraint(model.T, rule=energy_balance_grid)    
    
    # House energy balance
    def energy_balance_house(model, t, h):
        return model.PL1_SELL[t,h] - model.PL1_BUY[t,h] \
            == sum(model.generation[t,i] for i in [h] if h in model.G) \
            + sum(model.B_OUT[t,i] - model.B_IN[t,i] for i in [h] if h in model.B) \
            - sum(model.demand[t,i] for i in [h] if h in model.D)
    model.energy_balance_house = Constraint(model.T, model.H, rule=energy_balance_house)    

    # Battery energy balance
    def battery_soc(model, t, b):
        if t==model.T.at(1):
            return model.B_SOC[t,b] - model.battery_soc_ini[b] == model.battery_efficiency_charge[b]*model.B_IN[t,b]*model.dt  - (1/model.battery_efficiency_discharge[b])*model.B_OUT[t,b]*model.dt
        else:
            return model.B_SOC[t,b] - model.B_SOC[model.T.prev(t),b] == model.battery_efficiency_charge[b]*model.B_IN[t,b]*model.dt  - (1/model.battery_efficiency_discharge[b])*model.B_OUT[t,b]*model.dt
    model.battery_soc = Constraint(model.T, model.B, rule=battery_soc)    
    
    def battery_soc_fix(model, b):
        if value(model.battery_soc_fin[b]) > 0:
            return model.B_SOC[model.T.last(),b] == model.battery_soc_fin[b]
        else:
            return Constraint.Skip
    model.battery_soc_fix = Constraint(model.B, rule=battery_soc_fix)  
    
    return model


def get_results(solved_model):
    
    results = dict()
    results['community_members'] = solved_model.H.data()
    
    results['cost_energy'] = value(solved_model.COST_ENERGY[:])
    results['cost_grid'] = value(solved_model.COST_GRID[:])
    
    results['power_buy_community_level'] = value(solved_model.PL2_BUY[:])
    results['power_sell_community_level'] = value(solved_model.PL2_SELL[:])
    
    results['power_buy'] = {}
    results['power_sell'] = {}
    for m in results['community_members']:
        results['power_buy'][m] = value(solved_model.PL1_BUY[:,m])
        results['power_sell'][m] = value(solved_model.PL1_SELL[:,m])
    
    
    results['battery_soc'] = {}
    results['battery_charge'] = {}
    results['battery_discharge'] = {}
    for b in solved_model.B.data():
        results['battery_soc'][b] = value(solved_model.B_SOC[:,b])
        results['battery_charge'][b] = value(solved_model.B_IN[:,b])
        results['battery_discharge'][b] = value(solved_model.B_OUT[:,b])

    return results


def instantiate_model(model, model_data):

    model_instance = model.create_instance(model_data)

    return model_instance


def solve_model(model_instance, solver, tee = True, keepfiles=False):
    if 'path' in solver:
        optimizer = SolverFactory(solver['name'], executable=solver['path'])
    else:
        optimizer = SolverFactory(solver['name'])

    optimizer.solve(model_instance, tee=tee, keepfiles=keepfiles)

    return model_instance