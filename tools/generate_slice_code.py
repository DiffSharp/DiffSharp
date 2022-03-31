def generate(num_dims, fixed_dims):
    str = '[<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]'
    str += '\nmember t.GetSlice('
    prefix = ''
    for i in range(num_dims):
        if fixed_dims[i]:
            str += '{}i{}:int'.format(prefix, i)
        else:
            str += '{}i{}min:int option, i{}max:int option'.format(prefix, i, i)
        prefix = ', '
    str += ') ='
    str += '\n    // Dims: {}'.format(num_dims)
    # str += '\n    // Specified dims: {}'.format(fixed_dims)
    for i in range(num_dims):
        if fixed_dims[i]:
            str += '\n    let i{}given = 1'.format(i)
            str += '\n    let i{}min   = i{}'.format(i, i)
            str += '\n    let i{}max   = i{}'.format(i, i)
        else:
            str += '\n    let i{}given = if i{}min.IsSome && i{}max.IsSome then 1 else 0'.format(i, i, i)
            str += '\n    let i{}min   = defaultArg i{}min 0'.format(i, i)
            str += '\n    let i{}max   = defaultArg i{}max (t.shape[{}] - 1)'.format(i, i, i)
    str += '\n    let bounds = array2D ['
    prefix = ''
    for i in range(num_dims):
        str += '{}[i{}min; i{}max; i{}given]'.format(prefix, i, i, i)
        prefix = '; '
    str += ']'
    str += '\n    t.GetSlice(bounds)\n'
    return str

def per(n):
    ret = []
    for i in range(1<<n):
        s=bin(i)[2:]
        s='0'*(n-len(s))+s
        ret.append(list(map(lambda x: bool(int(x)),list(s))))
    return ret

def main():
    num_dims = 6
    str = ''
    for i in range(1,num_dims+1):
        for p in per(i):
            str += generate(i, p)
    print(str)

if __name__ == '__main__':
    main()
