import os


def ListFiles(filepath , filetype , fullpath=False):
    '''
    # lists all files in a given folder path of a given file extentsion
    :param filepath: string of folder path
    :param filetype: filetype
    :param expression: 1 = gives out the list as file-paths ; 0 = gives out the list as file-names only
    :return: list of files
    :return: list of files
    '''
    file_list = []
    for file in os.listdir(filepath) :
        if file.endswith(filetype) :
            if fullpath:
                file_list.append(file)
            elif fullpath == True :
                file_list.append(os.path.join(filepath , file))
    return file_list