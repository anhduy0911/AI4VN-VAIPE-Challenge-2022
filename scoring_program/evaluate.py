
import os
import sys
import rrc_evaluation_funcs
from script import default_evaluation_params, validate_data, evaluate_method
import subprocess
from zipfile import ZipFile
from os.path import basename
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("Polygon2")

def zip_files(name, folder):
    if os.path.exists("{}/{}.zip".format(folder, name)):
        os.remove("{}/{}.zip".format(folder, name))
    flag = False
    with ZipFile('{}/{}.zip'.format(folder, name), 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith(".txt"):
                    filePath = os.path.join(folderName, filename)
                    zipObj.write(filePath, basename(filePath))
                    flag = True
    if flag:
        os.system("rm {}/*.txt".format(folder))
    else:
        print("Wrong file format! Please read the instruction carefully!")
        exit()

if __name__ == "__main__":
    [_, input_dir, output_dir] = sys.argv
    submission_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')
    zip_files('prediction', submission_dir)
    res_dict = rrc_evaluation_funcs.main_evaluation({'g': '{}/groundtruth.zip'.format(truth_dir), 's': '{}/prediction.zip'.format(submission_dir)}, default_evaluation_params, validate_data, evaluate_method)
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        output_file.write("CER: {:f}\n".format(round(res_dict['method']['cer'], 4)))