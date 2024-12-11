import pandas as pd
import arff


def load_data():
    """
    Load the data from the arff file
    :return: data
    """
    data_freq = arff.load('data/freMTPL2freq.arff')
    df_freq = pd.DataFrame(data_freq, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge",
                                               "BonusMalus", "VehBrand", "VehGas", "Density", "Region"])
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)

    data_sev = arff.load('data/freMTPL2sev.arff')
    df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])
    df_sev["IDpol"] = df_sev["IDpol"].astype(int)

    # Left join drops the rows with a claim amount but no information about the driver
    # We fill contracts with no claims with 0
    data_merged = pd.merge(df_freq, df_sev, on='IDpol', how='left').fillna({'ClaimAmount': 0})

    # We convert the columns to the correct data types
    data_merged["IDpol"] = data_merged["IDpol"].astype(int)
    data_merged["ClaimNb"] = data_merged["ClaimNb"].astype(int)
    data_merged["Exposure"] = data_merged["Exposure"].astype(float)
    data_merged["Area"] = data_merged["Area"].astype("category")
    data_merged["VehPower"] = data_merged["VehPower"].astype(int)
    data_merged["VehAge"] = data_merged["VehAge"].astype(int)
    data_merged["DrivAge"] = data_merged["DrivAge"].astype(int)
    data_merged["BonusMalus"] = data_merged["BonusMalus"].astype(int)
    data_merged["VehBrand"] = data_merged["VehBrand"].astype("category")
    data_merged["VehGas"] = data_merged["VehGas"].astype("category")
    data_merged["Density"] = data_merged["Density"].astype(int)
    data_merged["Region"] = data_merged["Region"].astype("category")
    data_merged["ClaimAmount"] = data_merged["ClaimAmount"].astype(float)

    # Some preprocessing
    # Clip density to 10000, Driver age to 90, VehAge to 30 and BonusMalus to 125
    data_merged["Density"] = data_merged["Density"].clip(upper=10000)
    data_merged["DrivAge"] = data_merged["DrivAge"].clip(upper=90)
    data_merged["VehAge"] = data_merged["VehAge"].clip(upper=30)
    data_merged["BonusMalus"] = data_merged["BonusMalus"].clip(upper=125)

    return data_merged


if __name__ == '__main__':
    data = load_data()
    print(data.head())
