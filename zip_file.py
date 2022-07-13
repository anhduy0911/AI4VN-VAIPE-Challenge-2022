import os
import sys
# import rrc_evaluation_funcs
# from script import default_evaluation_params, validate_data, evaluate_method
import subprocess
from zipfile import ZipFile
from os.path import basename

def zip_files(name, folder):
    if os.path.exists("{}/{}.zip".format(folder, name)):
        os.remove("{}/{}.zip".format(folder, name))
    flag = False
    with ZipFile('{}/{}.zip'.format(folder, name), 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith(".csv"):
                    filePath = os.path.join(folderName, filename)
                    zipObj.write(filePath, basename(filePath))
                    flag = True
    # if flag:
    #     os.system("rm {}/*.txt".format(folder))
    # else:
    #     print("Wrong file format! Please read the instruction carefully!")
    #     exit()

if __name__ == '__main__':
    zip_files('groundtruth', 'data/private')
    zip_files('groundtruth', 'data/public')