"""
Core classes and functions. 
"""
import time
import contextlib

from collections import OrderedDict
from csb.core import validatedproperty

def format_time(t):

    units = [(1.,'s'),(1e-3,'ms'),(1e-6,'us'),(1e-9,'ns')]
    for scale, unit in units:
        if t > scale or t==0: break
        
    return '{0:.1f} {1}'.format(t/scale, unit)

@contextlib.contextmanager
def take_time(desc):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print('{0} took {1}'.format(desc, format_time(dt)))

    
class ctypeproperty(property):
    """
    Property decorator for ctype attributes. Acts as a proxy to the self.ctype
    object. The provided function will be used for validation with the setter.

    Example 1:

        >>> @ctypeproperty(float)
        >>> def b():
        >>>     pass

    Example 2:

        >>> @ctypeproperty
        >>> def a(v):
        >>>     v = int(v)
        >>>     if v < 0:
        >>>         raise ValueError(v)
        >>>     return v

    """
    def __init__(self, cast):
        self.cast = cast
        self(cast)

    def __call__(self, func):
        self.name = func.__name__
        self.__doc__ = func.__doc__
        return self

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return getattr(instance.ctype, self.name)

    def __set__(self, instance, value):
        setattr(instance.ctype, self.name, self.cast(value))

        
def get_properties(instance):
    return [attr for attr in instance.__class__.__dict__
            if isinstance(getattr(instance.__class__, attr), property)]


class CWrapper(object):

    def init_ctype(self):
        raise NotImplementedError

    def set_default_values(self):
        pass

    def __getstate__(self):

        state = OrderedDict()

        for attr in self.__dict__.keys() + get_properties(self):
            if attr == 'ctype': continue
            state[attr] = getattr(self, attr)

        return state

    def __setstate__(self, state):

        self.init_ctype()
        self.set_default_values()

        for attr in state:
            setattr(self, attr, state[attr])

            
class Nominable(object):
    """
    Mixin for 'thing with a name'
    """
    @validatedproperty
    def name(name):
        if name is not None:
            name = str(name)
        return name

    def __str__(self):
        return '{0}("{1}")'.format(self.__class__.__name__, self.name)

