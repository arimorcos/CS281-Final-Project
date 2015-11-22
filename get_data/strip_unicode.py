import sys, getopt, re

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:f")
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)

    output_file = []
    print_finds = False
    for opt, arg in opts:
        if opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_file = arg
        elif opt == "-f":
            print_finds = True

    matching_strings = []

    # read in each line
    with open(input_file, 'r') as f_in:
        f_out = open(output_file, 'w')

        for line in f_in:

            if not print_finds:

                line = line.replace("\u201c", "'")
                line = line.replace("\u201d", "'")
                line = line.replace("\u00ed", 'i')
                line = line.replace("\u00fa", 'u')
                line = line.replace("\u2019", "'")
                line = line.replace("\u00c1", "A")
                line = line.replace("\u00e9", "e")
                line = line.replace("\u2014", "-")
                line = line.replace("\u00eb", "e")
                line = line.replace("\u00e4", "a")
                line = line.replace("\u00f4", "o")
                line = line.replace("\u00f1", "n")
                line = line.replace("\u00e7", "c")
                line = line.replace("\u00fc", "u")
                line = line.replace("\u00da", "U")
                line = line.replace("\u00e1", "a")
                line = line.replace("\u00ef", "i")
                line = line.replace("\u2018", "'")
                line = line.replace("\u2026", "...")
                line = line.replace("\u00e8", "e")
                line = line.replace("\u010d", "c")
                line = line.replace("\u00e5", "a")
                line = line.replace("\u00e6", "ae")
                line = line.replace("\u00e0", "a")
                line = line.replace("\u00e2", "a")
                line = line.replace("\u00a0", " ")
                line = line.replace("\u00fb", "u")
                line = line.replace("\u00fd", "y")
                line = line.replace("\u00c9", "e")
                line = line.replace("\u00c6", "AE")
                line = line.replace("\u012b", "i")
                line = line.replace("\u010c", "C")
                line = line.replace("\u015a", "S")
                line = line.replace("\u00d8", "O")
                line = line.replace("\u015f", "s")
                line = line.replace("\u017b", "Z")
                line = line.replace("\u017c", "z")
                line = line.replace("\u017e", "z")
                line = line.replace("\u0142", "l")
                line = line.replace("\u017c", "z")
                line = line.replace("\ufffd", "")
                line = line.replace("\u017c", "z")
                line = line.replace("\u0144", "n")
                line = line.replace("\u0107", "c")
                line = line.replace("\u0160", "s")
                line = line.replace("\u0161", "s")
                line = line.replace("\u017c", "z")
                line = line.replace("\u00f3", "o")
                line = line.replace("\u00f2", "o")
                line = line.replace("\u00f5", "o")
                line = line.replace("\u00f6", "o")
                line = line.replace("\u00f8", "o")
                line = line.replace("\u00d1", "n")
                line = line.replace("\u00d6", "O")
                line = line.replace("\u0100", "A")
                line = line.replace("\u0101", "A")
                line = line.replace("\u00ee", "i")
                line = line.replace("\u011f", "g")
                line = line.replace("\u00ea", "e")
                line = line.replace("\u00ec", "i")
                line = line.replace("\u014c", "O")
                line = line.replace("\u014d", "o")
                line = line.replace("\u016b", "u")
                line = line.replace("\u0153", "oe")
                line = line.replace("\u0152", "OE")
                line = line.replace("\u0159", "r")
                line = line.replace("\u0130", "I")
                line = line.replace("\u0131", "i")
                line = line.replace("\u00e3", "a")
                line = line.replace("\u0129", "i")
                line = line.replace("\u2212", "-")
                line = line.replace("\u2212", "-")
                line = line.replace("\u2022", " ")
                line = line.replace("\u2261", "identical to")
                line = line.replace("\u0113", "e")
                line = line.replace("\u00f9", "u")
                line = line.replace("\u055a", "'")
                line = line.replace("\u22a6", "")
                line = line.replace("\u00dc", "U")
                line = line.replace("\u00b4", "'")
                line = line.replace("\u00b7", "-")
                line = line.replace("\u00b0", " degrees")
                line = line.replace("\u00c0", "A")
                line = line.replace("\u015b", "s")
                line = line.replace("\u0119", "e")
                line = line.replace("\u039b", "Lambda")
                line = line.replace("\u00d3", "O")
                line = line.replace("\u00bd", "one-half")
                line = line.replace("\u00ce", "I")
                line = line.replace("\u00be", "three-quarters")
                line = line.replace("\u0323", ".")
                line = line.replace("\u01f5", "g")
                line = line.replace("\u0148", "n")
                line = line.replace("\u00a3", "#")
                line = line.replace("\u2013", "-")
                line = line.replace("\u011b", "e")
                line = line.replace("\u00d7", "x")





                # write
                f_out.write(line)

            else:
                temp_match_strings = list(set(re.findall("(\\\u\w{4})", line)))
                matching_strings += temp_match_strings

    # remove duplicates
    matching_strings = list(set(matching_strings))
    if print_finds:
        for match in matching_strings:
            f_out.write(match + "\n")

    f_in.close()
    f_out.close()




