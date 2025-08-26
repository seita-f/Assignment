import numpy as np
import pandas as pd
import catboost as cb


def predict_for_dataset(
    df, features_df, prev_day_df,
    first_date, last_date,
    update_features_data,
    models,
    cat_features,
    location_columns=["Country/Region", "Province/State"]
):
    df['PredictedLogNewConfirmedCases'] = np.nan
    df['PredictedLogNewFatalities'] = np.nan
    df['PredictedConfirmedCases'] = np.nan
    df['PredictedFatalities'] = np.nan

    for day in pd.date_range(first_date, last_date):
        day_df = df[df['Date'] == day]
        if day_df.empty:
            continue

        day_features_pool = cb.Pool(features_df.loc[day_df.index], cat_features=cat_features)

        # predict LogNew* data
        for prediction_type in ['LogNewConfirmedCases', 'LogNewFatalities']:
            df.loc[day_df.index, 'Predicted' + prediction_type] = np.maximum(
                models[prediction_type].predict(day_features_pool),
                0.0
            )

        day_predictions_df = df.loc[day_df.index][
            location_columns + ['PredictedLogNewConfirmedCases', 'PredictedLogNewFatalities']
        ]

        # update Predicted ConfirmedCases and Fatalities
        for field in ['ConfirmedCases', 'Fatalities']:
            prev_day_field = field if day == first_date else ('Predicted' + field)
            merged_df = day_predictions_df.merge(
                right=prev_day_df[location_columns + [prev_day_field]],
                how='inner',
                on=location_columns
            )

            df.loc[day_df.index, 'Predicted' + field] = merged_df.apply(
                lambda row: row[prev_day_field] + np.rint(np.expm1(row['PredictedLogNew' + field])),
                axis='columns'
            ).values

        if update_features_data:
            # fill time delay embedding features based on this day for next days
            for next_day in pd.date_range(day + pd.Timedelta(days=1), last_date):
                next_day_features_df = features_df[df['Date'] == next_day]
                if next_day_features_df.empty:
                    continue

                merged_df = next_day_features_df[location_columns].merge(
                    right=day_predictions_df,
                    how='inner',
                    on=location_columns
                )

                prev_day_idx = (next_day - day).days
                for prediction_type in ['LogNewConfirmedCases', 'LogNewFatalities']:
                    features_df.loc[next_day_features_df.index, prediction_type + '_prev_day_%s' % prev_day_idx] = (
                        merged_df['Predicted' + prediction_type].values
                    )

        prev_day_df = df.loc[day_df.index]

    return df
