local num_states = 15;

local fst_tokenizer_path = "unicode_char_tokenizer_ipa.json";

local train_data_path = "data/pretrain_2isl/train_pretrain_s4_canonical.jsonl";
local dev_data_path = "data/pretrain_2isl/dev_pretrain_s4_canonical.jsonl";
local easy_dev_data_path = "data/pretrain_2isl/easy_dev_pretrain_s4_canonical.jsonl";
local test_data_path = "data/pretrain_2isl/test_pretrain_s4_canonical.jsonl";


local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
                        pretrained_model_name_or_path: "google/byt5-small"
                                };
                                
local data_loader(fname, batch_size) = {
        "f": "load_fst_jsonl",
        "batch_size": batch_size,
        "path": fname,
        "tokenizer": tokenizer,
        "num_states": num_states,
        "fst_tokenizer": fst_tokenizer_path,
        "fst_format": "isl_canon",
} ;


{
  "imports": ["import transformers", "from sip.data_loading import *", "from sip.fst_pretrain import *", "from sip.pretraining import *"],
  "logger": {
    f: "NeptuneLogger.create",
    "project": "sip-isl-fork/sip-isl"
  },
  "steps": [

   {
    "name": "pretrain_2isl_canon",
    "f": "pretrain",
    "model": {
        "f": "create_fst_pretraining_model",

        "machine_embedder": {
                "[lazy]": "create_simple_fst_embedder",
                "num_states": num_states,
                "state_embedding_dim": 64,
                "token_embedding_dim": 256,
                "final_state_embedding_dim": 16,
                "fst_tokenizer": fst_tokenizer_path,
                "fst_format": "isl_canon",
        },

        "model": {
            f: "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            pretrained_model_name_or_path: "google/byt5-small"
            },
    },

    "tokenizer": tokenizer,

    "train_data_loader": data_loader(train_data_path, 10),
    "easy_validation_data_loader": data_loader(easy_dev_data_path, 32),
    "validation_data_loader": data_loader(dev_data_path, 32),

    "test_data_loader": data_loader(test_data_path, 32),

    "optimizer": {"[lazy]": "torch.optim.Adam", "lr": 5e-4},
    "num_epochs": 20, #TODO

    "logger": "[logger]",

    "num_accumulation_steps": 3,

    "save_dir": "models/w_fsts_pretrain_s4_32",

    "train_data_path": train_data_path


   }

   ]
}
