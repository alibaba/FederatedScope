Here we give some hand-on examples, and roughly show their characteristics.

For more details, please see the specific configuration in each file.

| Yaml                                                | Dataset        | Model _type           | Algo     | Protect_method | Eval_protection |
| --------------------------------------------------- | -------------- | --------------------- | -------- | -------------- | --------------- |
| `gbdt_feature_gathering_on_abalone.yaml`            | Abalone (reg.) | `'feature_gathering'` | `'gbdt'` | None           | None            |
| `gbdt_feature_gathering_on_adult.yaml`              | Adult (clas.)  | `'feature_gathering'` | `'gbdt'` | None           | None            |
| `gbdt_label_scattering_on_adult.yaml`               | Adult (clas.)  | `'label_scattering'`  | `'gbdt'` | `'he'`         | None            |
| `rf_feature_gathering_on_abalone.yaml`              | Abalone (reg.) | `'feature_gathering'` | `'rf'`   | None           | None            |
| `rf_feature_gathering_on_adult.yaml`                | Adult (clas.)  | `'feature_gathering'` | `'rf'`   | None           | None            |
| `rf_label_scattering_on_adult.yaml`                 | Adult (clas.)  | `'label_scattering'`  | `'rf'`   | `'he'`         | None            |
| `xgb_feature_gathering_on_abalone.yaml`             | Abalone (reg.) | `'feature_gathering'` | `'xgb'`  | None           | None            |
| `xgb_feature_gathering_on_adult.yaml`               | Adult (clas.)  | `'feature_gathering'` | `'xgb'`  | None           | None            |
| `xgb_feature_gathering_dp_on_abalone.yaml`          | Abalone (reg.) | `'feature_gathering'` | `'xgb'`  | None           | None            |
| `xgb_feature_gathering_dp_on_adult.yaml`            | Adult (clas.)  | `'feature_gathering'` | `'xgb'`  | `'dp'`         | None            |
| `xgb_feature_gathering_op_boost_on_adult.yaml`      | Adult (clas.)  | `'feature_gathering'` | `'xgb'`  | `'op_boost'`   | None            |
| `xgb_label_scattering_on_abalone.yaml`              | Abalone (reg.) | `'label_scattering'`  | `'xgb'`  | `'he'`         | None            |
| `xgb_label_scattering_on_adult.yaml`                | Adult (clas.)  | `'label_scattering'`  | `'xgb'`  | `'he'`         | None            |
| `xgb_feature_gathering_dp_on_adult_by_he_eval.yaml` | Adult (clas.)  | `'feature_gathering'` | `'xgb'`  | `'he'`         | `'he'`          |