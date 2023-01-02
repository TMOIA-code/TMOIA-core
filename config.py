config = {
        "Transformer_output": 1,
        "Transformer_num_encoder": 3,
        # "Transformer_num_encoder": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),

        "epochs": 140,
        # "epochs": tune.grid_search([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        "learning_rate": 0.00001,
        # "learning_rate": tune.grid_search([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]),
        "batch_size_tr": 32,
        "batch_size_te": 32,
        "Transformer_drop": 0.1,
        # "Transformer_drop": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),

        "snp_input_dim": 2000,

        "Transformer_mRNA_input_dim": 2000,
        "Transformer_mRNA_hidden": 500,
        "Transformer_mRNA_head": 10,

        "Transformer_mCG_input_dim": 2000,
        "Transformer_mCG_hidden": 500,
        "Transformer_mCG_head": 10,

        # "Transformer_miRNA_drop": 0.5,
        "Transformer_mCHG_input_dim": 2000,
        "Transformer_mCHG_hidden": 500,
        "Transformer_mCHG_head": 10,

        "Transformer_mCHH_input_dim": 2000,
        "Transformer_mCHH_hidden": 500,
        "Transformer_mCHH_head": 10,

        "Transformer_snp_input_dim": 2000,
        "Transformer_snp_hidden": 500,
        "Transformer_snp_head": 10,

        # "Transformer_all_drop": 0.5,
        "Transformer_all_input_dim": 10000,
        "Transformer_all_hidden": 1000,
        "Transformer_all_head": 10,

        # "Transformer_integrate_drop": 0.5,
        "Transformer_integrate_input_dim": 2000,
        "Transformer_integrate_hidden": 1000,
        "Transformer_integrate_head": 10
}