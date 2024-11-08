tiny = {
    "n_agents": 2,
    "map_size": "tiny",
    "we": [1, 2.6],
    "inequality_mode": "tie",
    "max_steps": 500,
    "donation_capacity": 5,
    "survival_threshold": 10,
    "visual_radius": 0,
    "partial_observability": 0,
    "init_state": "empty",
    "reward_mode": "scalarised",
    "color_by_efficiency": True,
    "objective_order": "individual_first"
}
small = {
    "n_agents": 2,
    "map_size": "small",
    "we": [1, 2.6],
    "inequality_mode": "tie",
    "max_steps": 500,
    "donation_capacity": 5,
    "survival_threshold": 10,
    "visual_radius": 0,
    "partial_observability": 0,
    "init_state": "empty",
    "reward_mode": "scalarised",
    "color_by_efficiency": True,
    "objective_order": "individual_first"
}
medium = {
    "n_agents": 3,
    "map_size": "medium",
    "we": [1, 2.6],
    "inequality_mode": "loss",
    "max_steps": 500,
    "donation_capacity": 10,
    "survival_threshold": 15,
    "visual_radius": 4,
    "partial_observability": 1,
    "init_state": "empty",
    "efficiency": [0.2, 0.2, 0.67],
    "reward_mode": "scalarised",
    "color_by_efficiency": True,
    "objective_order": "individual_first"
}
large = {
    "n_agents": 5,
    "map_size": "large",
    "we": [1, 2.6],
    "inequality_mode": "loss",
    "max_steps": 500,
    "donation_capacity": 15,
    "survival_threshold": 30,
    "visual_radius": 4,
    "partial_observability": 1,
    "init_state": "empty",
    "efficiency": [0.2, 0.2, 0.67, 0.2, 0.67],
    "reward_mode": "scalarised",
    "color_by_efficiency": True,
    "objective_order": "individual_first"
}

very_large = {
    "n_agents": 10,
    "map_size": "very_large",
    "we": [1, 2.6],
    "inequality_mode": "loss",
    "max_steps": 500,
    "donation_capacity": 15,
    "survival_threshold": 30,
    "visual_radius": 4,
    "partial_observability": 1,
    "init_state": "empty",
    "efficiency": [0.5]*5+[0.15]*5,
    "reward_mode": "scalarised",
    "color_by_efficiency": True,
    "objective_order": "individual_first"
}
