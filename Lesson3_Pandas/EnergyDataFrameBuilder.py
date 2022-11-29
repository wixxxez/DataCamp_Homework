from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import EnergyDecoratorFactory
import NormalizeEnergyDataset
class Builder(ABC):



    @abstractmethod
    def energy_supply(self):
        pass

    @abstractmethod
    def energy_per_capita(self):

        pass

    @abstractmethod
    def remove_numbers_from_country_name(self):
        pass

    @abstractmethod
    def rename_country_name(self):

        pass

    @abstractmethod
    def remove_brackets(self):

        pass

class EnergyDataFrameBuilder(Builder):

    def __init__(self):

        self.reset()

    def reset(self):

        self._df = EmptyEnergyDataFrame()

    def get_df(self):

        df = self._df
        self.reset();
        return df;

    def energy_supply(self):

        self._df.add_decorator(EnergyDecoratorFactory.EnergySupplyDecoratorFactory())

    def energy_per_capita(self):
        self._df.add_decorator(EnergyDecoratorFactory.EnergyPerCapitaDecoratorFactory())

    def remove_brackets(self):

        self._df.add_decorator(EnergyDecoratorFactory.RemoveBracketsDecoratorFactory())

    def rename_country_name(self):

        self._df.add_decorator(EnergyDecoratorFactory.RenameCountryNameDecoratorFactory())

    def remove_numbers_from_country_name(self):

        self._df.add_decorator(EnergyDecoratorFactory.RemoveNumbersDecoratorFactory())

class EmptyEnergyDataFrame():

    def __init__(self):

        self._df = NormalizeEnergyDataset.EnergyDataFrame()

    def add_decorator(self, decorator):

        self._df = decorator.factory(self._df);

    def getDataFrameObject(self):
        return self._df;


class EnergyDataframeDirector():

    def __init__(self, builder):

        self.builder = builder;

    def BuildEnergyDataFrame(self):

        self.builder.energy_supply();
        self.builder.energy_per_capita()
        self.builder.remove_numbers_from_country_name()
        self.builder.remove_brackets()
        self.builder.rename_country_name()


def GetEnergyDataFrame():

    builder = EnergyDataFrameBuilder()
    director =EnergyDataframeDirector(builder)

    director.BuildEnergyDataFrame()
    df = builder.get_df()
    energy = df.getDataFrameObject().get_df()

    return energy
GetEnergyDataFrame()