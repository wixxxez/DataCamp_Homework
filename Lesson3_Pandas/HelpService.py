import pandas as pd

def get_top15withPopulations(top15):
    populations = [];
    for i in range(len(top15)):
        EnergySupply = top15.iloc[i]["Energy Supply"]
        EnergySupplyPerCapita = top15.iloc[i]["Energy Supply per capita"];
        populations.append(EnergySupply / EnergySupplyPerCapita)
    top15["Populations"] = populations

    return top15;
