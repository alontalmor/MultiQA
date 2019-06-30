import argparse
import json
import os
import boto3
import re
import zipfile
from  datasets.multiqa_factory import MultiQAFactory
from common.preprocess import MultiQAPreProcess
def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("dataset_name", type=str, help="use the actual name of the dataset class, case sensitive")
    parse.add_argument("split", type=str, help="dev / train / text")
    parse.add_argument("output_file", type=str, help="")
    parse.add_argument("--sample_size", type=int, help="", default=None)
    parse.add_argument("--n_processes", type=int, help="", default=1)
    args = parse.parse_args()

    preprocessor = MultiQAPreProcess(args.n_processes)
    factory = MultiQAFactory()
    header, contexts = factory.build_dataset(args.dataset_name, args.split, preprocessor, args.sample_size)

    if args.output_file.startswith('s3://'):
        output_file = args.output_file.replace('s3://','')
        bucketName = output_file.split('/')[0]
        outPutname = '/'.join(output_file.split('/')[1:])
        local_filename = outPutname.replace('/','_')
        with open(local_filename.replace('.zip',''), "w") as f:
            # first JSON line is header
            f.write(json.dumps({'header':header}) + '\n')
            for instance in contexts:
                f.write(json.dumps(instance) + '\n')

        with zipfile.ZipFile(local_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(local_filename.replace('.zip',''))

        s3 = boto3.client('s3')
        s3.upload_file(local_filename , bucketName, outPutname, ExtraArgs={'ACL':'public-read'})

        os.remove(local_filename)
        os.remove(local_filename.replace('.zip',''))
    else:
        output_dir = '/'.join(args.output_file.split('/')[0:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output_file.replace('.zip',''), "w") as f:
            # first JSON line is header
            f.write(json.dumps({'header': header}) + '\n')
            for instance in contexts:
                if True:
                    s = json.dumps(instance, sort_keys=True, indent=4)
                    # just making the answer starts in the sample no have a newline for every offset..
                    s = re.sub('\n\s*(\d+)', r'\1', s)
                    #s = re.sub('\n\s*"title"', r'"title"', s)
                    s = re.sub('(\d+)\n\s*]', r'\1]', s)
                    s = re.sub('(\d+)],\n\s*', r'\1],', s)
                    s = re.sub('\[\s*\n', r'[', s)
                    s = re.sub('\[\s*', r'[', s)
                    #s = re.sub('","([\_a-zA-Z]+)":', r'",\n"\1":', s)
                    f.write(s + '\n')
                else:
                    f.write(json.dumps(instance) + '\n')

        if args.output_file.endswith('zip'):
            with zipfile.ZipFile(args.output_file, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.write(args.output_file.replace('.zip',''))

            os.remove(args.output_file.replace('.zip',''))

if __name__ == '__main__':
    main()