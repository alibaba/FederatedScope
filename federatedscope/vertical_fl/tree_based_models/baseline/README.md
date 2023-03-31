Here we give some hand-on examples, and roughly show their characteristics.

For more details, please see the specific configuration in each yaml files.

| Yaml                                                | Task  | Algo     | Protect_method | Eval_protection |
| --------------------------------------------------- | ----- | -------- | -------------- | --------------- |
| `gbdt_feature_gathering_on_abalone.yaml`            | Reg.  | `'gbdt'` | None           | None            |
| `gbdt_feature_gathering_on_adult.yaml`              | Clas. | `'gbdt'` | None           | None            |
| `gbdt_label_scattering_on_adult.yaml`               | Clas. | `'gbdt'` | `'he'`         | None            |
| `rf_feature_gathering_on_abalone.yaml`              | Reg.  | `'rf'`   | None           | None            |
| `rf_feature_gathering_on_adult.yaml`                | Clas. | `'rf'`   | None           | None            |
| `rf_label_scattering_on_adult.yaml`                 | Clas. | `'rf'`   | `'he'`         | None            |
| `xgb_feature_gathering_on_abalone.yaml`             | Reg.  | `'xgb'`  | None           | None            |
| `xgb_feature_gathering_on_adult.yaml`               | Clas. | `'xgb'`  | None           | None            |
| `xgb_feature_gathering_dp_on_abalone.yaml`          | Reg.  | `'xgb'`  | None           | None            |
| `xgb_feature_gathering_dp_on_adult.yaml`            | Clas. | `'xgb'`  | `'dp'`         | None            |
| `xgb_feature_gathering_op_boost_on_adult.yaml`      | Clas. | `'xgb'`  | `'op_boost'`   | None            |
| `xgb_label_scattering_on_abalone.yaml`              | Reg.  | `'xgb'`  | `'he'`         | None            |
| `xgb_label_scattering_on_adult.yaml`                | Clas. | `'xgb'`  | `'he'`         | None            |
| `xgb_feature_gathering_dp_on_adult_by_he_eval.yaml` | Clas. | `'xgb'`  | `'he'`         | `'he'`          |