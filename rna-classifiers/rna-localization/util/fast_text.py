import os
import fastText

def bin_to_vec(modelFilePath):
    # Load model
    model = fastText.load_model(modelFilePath)

    # Create file
    words = model.get_words()
    with open(os.path.splitext(modelFilePath)[0] + ".vec", 'w') as f:
        # First line
        f.write(str(len(words)) + " " + str(model.get_dimension()) + "\n") 

        # Words and vectors
        for w in words:
            v = model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                f.write(w + vstr + "\n")
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass

def train_supervised(
    input,
    lr=0.1,
    dim=100,
    ws=5,
    epoch=5,
    minCount=1,
    minCountLabel=0,
    minn=0,
    maxn=0,
    neg=5,
    wordNgrams=1,
    loss="softmax",
    bucket=2000000,
    thread=12,
    lrUpdateRate=100,
    t=1e-4,
    label="__label__",
    verbose=2,
    pretrainedVectors="",
    ):

    # Training
    if pretrainedVectors != "":
        # C++ script

        # Create command
        input_to_dict = {
            "input": input,
            "lr": lr,
            "dim": dim,
            "ws": ws,
            "epoch": epoch,
            "minCount": minCount,
            "minCountLabel": minCountLabel,
            "minn": minn,
            "maxn": maxn,
            "neg": neg,
            "wordNgrams": wordNgrams,
            "loss": loss,
            "bucket": bucket,
            "thread": thread,
            "lrUpdateRate": lrUpdateRate,
            "t": t,
            "label": label,
            "verbose": verbose,
            "pretrainedVectors": pretrainedVectors
        }
        model_path = os.environ['MY_ROOT_APPLICATION']+"/rna-localization/model/model_temp"
        cmd_str = os.environ['FASTTEXT_BIN_PATH']+" supervised -output "+model_path
        for k in input_to_dict:
            cmd_str += " -" + str(k) + " " + str(input_to_dict[k])
        
        os.system(cmd_str)
        model = fastText.load_model(model_path+".bin")

        # Delete model temp
        os.remove(model_path+".bin")
        os.remove(model_path+".vec")

        return model
    else:
        # Python
        return fastText.train_supervised(
            input,
            lr=lr,
            dim=dim,
            ws=ws,
            epoch=epoch,
            minCount=minCount,
            minCountLabel=minCountLabel,
            minn=minn,
            maxn=maxn,
            neg=neg,
            wordNgrams=wordNgrams,
            loss=loss,
            bucket=bucket,
            thread=thread,
            lrUpdateRate=lrUpdateRate,
            t=t,
            label=label,
            verbose=verbose
            )


