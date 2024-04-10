import yaml
from yaml import Loader
#from dataclasses import dataclass

class Group:
    def __init__(self, contents):
        self._retrievables = {}
        if isinstance(contents, dict):
            for k,v in contents.items():
                if isinstance(v, dict):
                    v = Group(v)
                if isinstance(k, int):
                    self._retrievables[k] = v
                elif isinstance(k, str):
                    setattr(self,k,v)
        elif isinstance(contents, list):
            self.values = []
            for v in contents:
                if isinstance(v, dict):
                    v = Group(v)
                else:
                    self.values.append(v)
    
    def __repr__(self):
        return self.values
    
    def __getitem__(self, key):
        if key in self._retrievables.keys():
            return self._retrievables[key]
        else:
            return getattr(self,key)

#class Field:
#    def __init__(self, value):
#        self.value = value
#    def __get__(self):
#        return self.value

class Config:
    def __init__(self, f):
        raw = yaml.load(f, Loader=Loader)
        self.groups = []
        for k,v in raw.items():
            if isinstance(v, dict) or isinstance(v, list):
                v = Group(v)
                self.groups.append(k)
            else:
                v = Field(v)
            setattr(self,k,v)

if __name__ == '__main__':
    fp = r"F:/repos/M3CV/templates/config_outcome_prediction.yaml"
    with open(fp, 'r') as f:
        config = Config(f)