from dataclasses import dataclass, field
import typing as tp


class Registry:
    def __init__(self):
        self._factory_classes = {}
        self._configs = {}
        self._names_list = []
        self._premade_factory_functions = {}

    @dataclass
    class BaseConfig:
        """
        Utility class to inherit from, useful for
        simple-parsing subgroups
        https://github.com/lebrice/SimpleParsing/tree/master/examples/subgroups
        """
        name = "base"

    def register_factory(self, factory):
        """
        Decorates a factory to register it within the registry. This method allows the
        class to be assigned with a name, and config object that can be used to specify the construction.
        if used, the class being decorated should include an inner "Config"
        dataclass inheriting from BaseConfig, and a __call__ method
        which builds the desired object given a config.
        name to the factory class. After being decorated, the class and config will be registered
        with this class, and can be accessed via .build(name, cfg) and
        .get_config(name).
        """
        if type(factory) == type: 
            # we are dealing with a class
            assert hasattr(factory, "Config"), "You need to define a config class"
            config = factory.Config
            name = config.name
            if config.name == 'premade': 
                raise ValueError('The factory name `premade` is reserved. Choose another name!')
            self._factory_classes[name] = factory()
            self._configs[name] = config
            self._names_list.append(name)

        else: 
            # we are dealing with a factory function, 
            # which should be registered under premades
            name = factory.__name__
            self._premade_factory_functions[name] = factory
        
        return factory

    def build(self, config, *args, **kwargs):
        if config.name not in self.list_constructibles():
            raise ValueError(f"Can't resolve the factory named {config.name}.")
        if config.name == 'premade':
            factory = self._premade_factory_functions[config.id]
            return factory(*args, **kwargs)
        else: 
            factory = self._factory_classes[config.name]
            return factory(config, *args, **kwargs)

    def list_constructibles(self):
        keys =  self._names_list.copy()
        if self._premade_ids: 
            keys = keys + ['premade'] 
        return keys

    def get_config(self, name):
        if name == 'premade': 
            return self.Premade
        else: 
            return self._configs[name]
    
    @property
    def _premade_ids(self): 
        return tuple(self._premade_factory_functions.keys())

    @property
    def Premade(self): 
        @dataclass
        class Premade: 
            name = 'premade'
            id: tp.Literal[self._premade_ids] = self._premade_ids[0]
        return Premade 