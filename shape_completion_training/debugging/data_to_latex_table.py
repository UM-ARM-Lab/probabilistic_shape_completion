data_5 = """
3DrecGAN++		7.3	27.2	2.9	9.6	4.6	12.4
Ours		5.2	19.9	4.4	4.5	2.3	8.9
VAE		6.3	24.7	2.8	7.8	3.0	10.6
VAE_GAN		6.4	25.2	2.7	7.9	3.1	10.6
"""

data_30 = """
3DrecGAN++		3.5	7.2	1.2	3.6	1.2	4.8
Ours		2.3	13.6	3.2	1.7	1.3	4.8
VAE		3.4	7.0	1.8	3.1	1.5	4.9
VAE_GAN		3.6	7.3	1.6	3.3	1.3	5.0
"""

name_map = {
    "Ours": "Ours",
    "VAE": "VAE",
    "3DrecGAN++": "3D-rec-GAN",
    "VAE_GAN": "VAE-GAN"
}


def parse_data(data):
    parsed = {}
    for i, row in enumerate(data.split('\n')):
        if row == '':
            continue
        cols = row.split('\t')
        method = cols[0]
        nums = [float(num) for num in cols[4:]]


        parsed[method] = " {:6.1f} & {:6.1f} & {:6.1f} & {:6.1f}   ".format(nums[2], nums[1], nums[0],
                                                                                  nums[3])
    return parsed


parsed_5 = parse_data(data_5)
parsed_30 = parse_data(data_30)

order = ["Ours", "VAE", "3DrecGAN++", "VAE_GAN"]
for method in order:
    print("{:<10} & {} &  {} \\\\".format(name_map[method], parsed_30[method], parsed_5[method]))
