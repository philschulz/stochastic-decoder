"""

"""
from argparse import ArgumentParser
import logging

def main():

    logging.basicConfig(level=logging.INFO)

    commandline_parser = ArgumentParser("A script that cleans parallel corpora. It removes lines starting "
                                        "with xml or html annotations as well as empty lines. The original "
                                        "files will be replaced by the cleaned files.")

    commandline_parser.add_argument("files", nargs=2,
                                    help="The two sides of the parallel corpus.")

    args = commandline_parser.parse_args()

    files = args.files
    file1 = files[0]
    file2 = files[1]

    logging.info("Reading files")
    lang1 = list()
    lang2 = list()
    with open(file1) as first, open(file2) as second:
        for line in first:
            lang1.append(line.strip())
        for line in second:
            lang2.append(line.strip())

    assert len(lang1) == len(lang2), "The input files contain different number of lines ({} vs {}).".format(len(lang1), len(lang2))

    logging.info("Processing files")
    out1 = list()
    out2 = list()
    for line1, line2 in zip(lang1,lang2):
        non_empty = len(line1) > 0 and len(line2) > 0
        non_annotated = not (line1.startswith('<') or line2.startswith('<'))
        if non_empty and non_annotated:
            out1.append(line1)
            out2.append(line2)

    logging.info("Writing output")
    with open(file1, "w") as outfile1, open(file2, "w") as outfile2:
        outfile1.write("\n".join(out1))
        outfile2.write("\n".join(out2))

    logging.info("Finished!")


if __name__ == "__main__":
    main()