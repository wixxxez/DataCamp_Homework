import NormalizeEnergyDataset
class Factory():

    def factory(self, df_obj):

        pass;

class EnergySupplyDecoratorFactory(Factory):

    def factory(self, df_obj):

        return NormalizeEnergyDataset.NormalizeEnergySupplyDecorator(df_obj)

class EnergyPerCapitaDecoratorFactory(Factory):

    def factory(self, df_obj):

        return NormalizeEnergyDataset.NormalizeEnergyPerCapitaDecorator(df_obj)

class RemoveNumbersDecoratorFactory(Factory):

    def factory(self, df_obj):

        return NormalizeEnergyDataset.RemoveNumbersFromCountryNameDecorator(df_obj)

class RemoveBracketsDecoratorFactory(Factory):

    def factory(self, df_obj):

        return  NormalizeEnergyDataset.RemoveTextInBracketsFromCountryName(df_obj)

class RenameCountryNameDecoratorFactory(Factory):

    def factory(self, df_obj):
        return NormalizeEnergyDataset.RenameCountriesDecorator(df_obj);
