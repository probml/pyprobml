def plotColors():

    colors = ['b', 'r', 'k', 'g', 'c', 'y',
              'm', 'r', 'b', 'k', 'g', 'c', 'y', 'm']
    symbols = ['o', 'x', '*', '>', '<', '^',
               'v', '+', 'p', 'h', 's', 'd', 'o', 'x']
    styles = ['-', ':', '-.', '--', '-', ':', '-.',
              '--', '-', ':', '-.', '--', '-', ':', '-.', '--']
    Str = []
    print(len(colors))
    for i in range(0, len(colors)):
        Str.append(colors[i] + styles[i])
    return [styles, colors, symbols, Str]


[styles, colors, symbols, Str] = plotColors()
print(Str)
