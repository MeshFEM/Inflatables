def write_line_mesh(filename, pts, edges):
    f = open(filename, 'w')
    for p in pts: f.write("v\t{}\t{}\t{}\n".format(*p))
    for e in edges: f.write("l\t{}\t{}\n".format(e[0] + 1, e[1] + 1))
