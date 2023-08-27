import os
import random
import pandas as pd
import argparse

def create_csv(args):

    zeros = os.listdir(args.zero_dir)
    ones = os.listdir(args.one_dir)
    data = []
    for z in zeros[:args.nbr]:
        data.append([os.path.join(args.zero_dir, z), 0])
    for o in ones:
        data.append([os.path.join(args.one_dir, o), 1])
    random.shuffle(data)

    df = pd.DataFrame(data, columns=["path", "class"])
    df.to_csv(f"{args.save_name}_data.csv", index=False)

def common_data_csv(args):

    common = pd.read_csv(os.path.join(args.zero_dir,"test.tsv"), sep="\t")
    common["class"] = 0
    common["path"] = common["path"].apply(lambda x: os.path.join(os.path.join(args.zero_dir,"clips"), x + ".mp3"))
    new_df = common[["path", "class"]][:1500]
    ones = os.listdir(args.one_dir)

    data = []
    for o in ones:
        data.append([os.path.join(args.one_dir, o), 1])

    df_wake = pd.DataFrame(data, columns=["path", "class"])
    df = pd.concat([df_wake, new_df], ignore_index=True)
    df = df.sample(frac=1)
    df.reset_index(drop=True)
    df.to_csv("test_data_wakeword.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-zdir", "--zero_dir", required=True,
                        help="the directory of the non wake word data.")
    parser.add_argument("-odir", "--one_dir", required=True,
                        help="the directory of the wake word data.")
    parser.add_argument("-sname", "--save_name", required=True,
                        help="the name of the csv file .ie train or test.")
    parser.add_argument("-nbr", "--nbr_instances", required=True,
                        help="the number of instance to select from zero dataset.")
    parser.add_argument("-c", "--choice", choices=[0,1], required=True,
                        help="if you are using common voice dataset use 1 else 0.")
    args = parser.parse_args()

    if args.choice == 0:
        create_csv(args)
    else: 
        common_data_csv(args)


    