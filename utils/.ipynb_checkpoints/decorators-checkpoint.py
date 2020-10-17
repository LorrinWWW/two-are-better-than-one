

def warn_not_override(fun):
    def inner(*args, **kargs):
        print(f'warning: this function [{fun}] may need to be overridden, otherwise the behavior is unpredictable.')
        return fun(*args, **kargs)
    return inner

def must_override(fun):
    def inner(*args, **kargs):
        raise Exception(f'this function [{fun}] must be overridden')
        return fun(*args, **kargs)
    return inner