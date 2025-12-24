from engine import Seq2SeqEngine
from models import Encoder, Decoder, Seq2Seq
from engine import Seq2SeqEngine
 
data_en = [
    "Foxes are adorable creatures.",
    "Arctic foxes are cute and adorable",
    "Gray foxes are adorable and cute.",
    "Foxes are cute",
    "Foxes are adorable",

    "Gray fox",
    "Arctic foxes",
    "The fennec fox is cute",
    "Fox is cute",
    "That fox is cute and adorable",

    "Foxes are small-to-medium-sized omnivorous mammals ",
    "Foxes are omnivores",
    "Bat-eared fox in Kenya",
    "Gray fox in Midtown, Palo Alto, California",
    "The fennec fox is the smallest species of fox",

]

data_de = [
    "Füchse sind bezaubernde Tiere.",
    "Polarfüchse sind süß und liebenswert.",
    "Graufüchse sind süß und liebenswert.",
    "Füchse sind süß.",
    "Füchse sind bezaubernd.",

    "Graufuchs.",
    "Polarfüchse.",
    "Der Wüstenfuchs ist süß.",
    "Fuchs ist süß.",
    "Dieser Fuchs ist süß und liebenswert.",

    "Füchse sind kleine bis mittelgroße Allesfresser.",
    "Füchse sind Allesfresser.",
    "Löffelhund in Kenia",
    "Graufuchs in Midtown, Palo Alto, Kalifornien",
    "Der Wüstenfuchs ist die kleinste Fuchsart.",
]


engine = Seq2SeqEngine(
    encoder_class=Encoder,
    decoder_class=Decoder,
    src_lang="en",
    tgt_lang="de",
    encoder_params={"embedding_size":32, "hidden_size":64, "num_layers":1, "p":0.0},
    decoder_params={"embedding_size":32, "hidden_size":64, "num_layers":1, "p":0.0},
    load_model=False
)

engine.train(data_en, data_de, epochs=100)
engine.evaluate(data_en, data_de)
print(engine.translate("Foxes are adorable creatures."))