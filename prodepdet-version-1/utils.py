import inspect
from inspect import iscoroutinefunction, isgeneratorfunction


def fix(args, kwargs, sig):
    ba = sig.bind(*args, **kwargs)
    ba.apply_defaults()
    return ba.args, ba.kwargs


def decorate(func, caller, extras=(), kwsyntax=False):
    sig = inspect.signature(func)
    if iscoroutinefunction(caller):
        async def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            return await caller(func, *(extras + args), **kw)
    elif isgeneratorfunction(caller):
        def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            for res in caller(func, *(extras + args), **kw):
                yield res
    else:
        def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            return caller(func, *(extras + args), **kw)
    fun.__name__ = func.__name__
    fun.__doc__ = func.__doc__
    __wrapped__ = func  # support nested wrap
    fun.__signature__ = sig
    fun.__qualname__ = func.__qualname__
    # builtin functions like defaultdict.__setitem__ lack many attributes
    try:
        fun.__defaults__ = func.__defaults__
    except AttributeError:
        pass
    try:
        fun.__kwdefaults__ = func.__kwdefaults__
    except AttributeError:
        pass
    try:
        fun.__annotations__ = func.__annotations__
    except AttributeError:
        pass
    try:
        fun.__module__ = func.__module__
    except AttributeError:
        pass
    try:
        fun.__dict__.update(func.__dict__)
    except AttributeError:
        pass
    fun.__wrapped__ = __wrapped__  # support nested wrap
    return fun
