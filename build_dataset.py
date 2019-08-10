import argparse
import json
import os
import boto3
import re
import gzip
import shutil
from datasets.multiqa_factory import MultiQAFactory
from common.preprocess import MultiQAPreProcess
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--dataset_name", type=str, help="use the actual name of the dataset class, case sensitive")
    parse.add_argument("--dataset_flavor", type=str, help="", default=None)
    parse.add_argument("--dataset_version", type=str, help="", default=None)
    parse.add_argument("--dataset_specific_props", type=str, help="", default='' , action = 'append')
    parse.add_argument("--split", type=str, help="dev / train / test")
    parse.add_argument("--output_file", type=str, help="")
    parse.add_argument("--header_file", type=str, help="If this file path is provided the header json will be save here.", default=None)
    parse.add_argument("--input_file", type=str, help="", default=None)
    parse.add_argument("--sample_size", type=int, help="", default=None)
    parse.add_argument("--save_in_sample_format", type=bool, help="", default=False)
    parse.add_argument("--n_processes", type=int, help="", default=1)
    args = parse.parse_args()

    preprocessor = MultiQAPreProcess(args.n_processes)
    factory = MultiQAFactory()
    header, contexts = factory.build_dataset(args.dataset_name, args.split, args.dataset_version, args.dataset_flavor, args.dataset_specific_props, \
                                             preprocessor, args.sample_size, args.input_file)
    print('------- dataset header --------')
    print(json.dumps({'header': header}, indent=4))


    if args.header_file is not None:
        print('writing header at %s' % args.header_file)
        if args.header_file.startswith('s3://'):
            header_file = args.header_file.replace('s3://', '')
            bucketName = header_file.split('/')[0]
            outPutname = '/'.join(header_file.split('/')[1:])

            with open('temp.json', "w") as f:
                json.dump(header,f, indent=4)
            s3 = boto3.client('s3')
            s3.upload_file('temp.json', bucketName, outPutname, ExtraArgs={'ACL': 'public-read'})
            os.remove('temp.json')
        else:
            with open(args.header_file, "w") as f:
                json.dump(header,f, indent=4)

    if args.output_file.startswith('s3://'):
        output_file = args.output_file.replace('s3://','')
        bucketName = output_file.split('/')[0]
        outPutname = '/'.join(output_file.split('/')[1:])
        local_filename = outPutname.replace('/','_')



        with gzip.open(local_filename, "wb") as f:
            # first JSON line is header
            f.write((json.dumps({'header':header}) + '\n').encode('utf-8'))
            for instance in contexts:
                f.write((json.dumps(instance) + '\n').encode('utf-8'))

        print("size of %s is %dMB" % (local_filename, int(os.stat(local_filename).st_size / 1000000)))

        s3 = boto3.client('s3')
        s3.upload_file(local_filename , bucketName, outPutname, ExtraArgs={'ACL':'public-read'})

        os.remove(local_filename)
    else:
        output_dir = '/'.join(args.output_file.split('/')[0:-1])
        if output_dir != '' and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output_file.replace('.gz',''), "w") as f:
            # first JSON line is header
            f.write(json.dumps({'header': header}, indent=4) + '\n')
            for instance in contexts:
                if args.save_in_sample_format:
                    s = json.dumps(instance, indent=4)
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

        if args.output_file.endswith('gz'):
            with open(args.output_file.replace('.gz',''), 'rb') as f_in:
                with gzip.open(args.output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(args.output_file.replace('.gz',''))

if __name__ == '__main__':
    main()