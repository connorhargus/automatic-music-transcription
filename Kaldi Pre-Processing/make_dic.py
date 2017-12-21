

"""
Create a simple dictionary file mapping each note to itself (phones to words)
"""
def make_dic():
    with open('db/TEDLIUM_release2/TEDLIUM.152k.dic', 'w') as dic_file:

        for i in range(21, 109):
            dic_file.write(str(i) + ' ' + str(i) + '\n')