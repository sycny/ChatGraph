from GPT_generation import Chatting
from tqdm import tqdm


if __name__ == "__main__":

    KEY = ""

    R8_classes = "[money-fx, trade, acq, grain, interest, crude, ship]"
#     _20NG_classes = "['sci.electronics', 'comp.windows.x', 'comp.os.ms-windows.misc', 'talk.politics.mideast', 'talk.politics.guns', 'rec.motorcycles', 'comp.graphics', 'rec.sport.baseball', 'comp.sys.mac.hardware', 'talk.politics.misc', 'misc.forsale', 'sci.space', 'talk.religion.misc', 'sci.med', 'rec.autos', 'sci.crypt', \
# 'comp.sys.ibm.pc.hardware', 'alt.atheism', 'soc.religion.christian', 'rec.sport.hockey]"
    R52_classes = "['grain', 'veg-oil', 'lead', 'sugar', 'crude', 'wpi', 'copper', 'heat', 'ship', 'instal-debt', 'livestock', 'zinc', 'cocoa', 'bop', 'lumber', 'platinum', 'rubber', 'carcass', 'orange', 'gold', 'gnp', 'iron-steel', 'reserves', 'gas', 'dlr', 'trade', 'money-fx', 'ipi', 'potato', 'cotton', 'alum', 'jobs', 'pet-chem', 'fuel', 'cpu', 'housing', 'acq', 'tea', 'earn', 'coffee', 'lei', 'meal-feed', 'jet', 'retail', 'cpi', 'nickel', 'money-supply', 'tin', 'nat-gas', 'strategic-metal', 'income', 'interest']"

    _20NG_dict = {0: "'sci.electronics'", 1: " 'comp.windows.x'", 2: " 'comp.os.ms-windows.misc'", 3: " 'talk.politics.mideast'", 4: " 'talk.politics.guns'", 5: " 'rec.motorcycles'", 6: " 'comp.graphics'", 7: " 'rec.sport.baseball'", 8: " 'comp.sys.mac.hardware'", 9: " 'talk.politics.misc'", 10: " 'misc.forsale'", 11: " 'sci.space'", 12: " 'talk.religion.misc'", 13: " 'sci.med'", 14: " 'rec.autos'", 15: " 'sci.crypt'", 16: " 'comp.sys.ibm.pc.hardware'", 17: " 'alt.atheism'", 18: " 'soc.religion.christian'", 19: " 'rec.sport.hockey"}
    _20NG_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    MR_classes = "[positive, negative]"

    ohsumed_classes = "['C15', 'C03', 'C01', 'C05', 'C10', 'C18', 'C16', 'C13', 'C23', 'C08', 'C22', 'C06', 'C02', 'C12', 'C14', 'C19', 'C11', 'C20', 'C21', 'C04', 'C17', 'C07', 'C09']"

#     instruction = f"You are a text classifier and your task is to classifiy a given text into the following categories: {_20NG_classes}. \
# If you cannot decide, just output None. You should just output the answer from the above categories. Do not output a sentence.\n\
# The meaning of each label is as follows: {_20NG_dict}"

#     instruction = f"You are a text classifier and your task is to classifiy a given text into the following categories: {_20NG_classes}. \
# If you cannot decide, just output None. You should just output the answer from the above categories. Do not output a sentence.\n\
# The meaning of each label is as follows: {_20NG_dict}\
# \nGood example:\n\
# ###Input:\ndecay cbnewsj cb att com \( dean kaflowitz \) subject bible quiz answers organization distribution na lines 18 article healta 153 saturn wwc edu , healta saturn wwc edu \( tammy r healy \) writes 12 \) 2 ark covenant god said make image , refering idols , created worshipped ark covenant n't high priest could enter holy kept year , day atonement familiar , knowledgeable original language , believe word idol translator would used word idol instead image original said idol think 're wrong , could suggesting way determine whether interpretation offer correct dean kaflowitz.\n###Output:\n17\n\
# \nBad example:\n\
# ###Input:\ndecay cbnewsj cb att com \( dean kaflowitz \) subject bible quiz answers organization distribution na lines 18 article healta 153 saturn wwc edu , healta saturn wwc edu \( tammy r healy \) writes 12 \) 2 ark covenant god said make image , refering idols , created worshipped ark covenant n't high priest could enter holy kept year , day atonement familiar , knowledgeable original language , believe word idol translator would used word idol instead image original said idol think 're wrong , could suggesting way determine whether interpretation offer correct dean kaflowitz.\n###Output:\n'alt.atheism'\n"

    instruction = f"You are a text classifier and your task is to classifiy a given text into the following categories: {ohsumed_classes}. \
If you cannot decide, just output None. You should just output the answer from the above categories. Do not output a sentence.\n\
\nGood example:\n\
###Input:\nbahia cocoa review continued throughout week bahia cocoa zone drought since early january improving prospects coming temporao although normal levels restored comissaria smith said weekly review dry period means temporao late year arrivals week ended february bags kilos making cumulative total season mln stage last year seems cocoa delivered earlier included arrivals figures comissaria smith said still doubt much old crop cocoa still available harvesting come end total bahia crop estimates around mln bags sales standing almost mln hundred thousand bags still hands farmers exporters processors doubts much cocoa would fit export shippers obtaining bahia superior certificates view lower quality recent weeks farmers sold good part cocoa held comissaria smith said spot bean prices rose cruzados per kilos bean shippers reluctant offer nearby shipment limited sales booked march shipment dlrs per tonne ports named new crop sales also light open ports june july going dlrs dlrs new york july aug sept dlrs per tonne fob routine sales butter made march april sold dlrs april may butter went times new york may june july dlrs aug sept dlrs times new york sept oct dec dlrs times new york dec comissaria smith said destinations u currency areas uruguay open ports sales registered dlrs march april dlrs may dlrs aug times new york dec oct dec buyers u argentina uruguay convertible currency areas liquor sales limited march april selling dlrs june july dlrs times new york july aug sept dlrs times new york sept oct dec times new york dec comissaria smith said total bahia sales currently estimated mln bags crop mln bags crop final figures period february expected published brazilian cocoa trade commission ends midday february reuter.\n###Output:\ncocoa\n\
\nGood example:\n\
###Input:\nchampion products approves stock split champion products inc said board directors approved two one stock split common shares shareholders record april company also said board voted recommend shareholders annual meeting april increase authorized capital stock five mln mln shares reuter.\n###Output:\nearn\n\
\nGood example:\n\
###Input:\ncomputer terminal systems completes sale computer terminal systems inc said completed sale shares common stock warrants acquire additional one mln shares n v switzerland dlrs company said warrants exercisable five years purchase price dlrs per share computer terminal said also right buy additional shares increase total holdings pct computer terminal outstanding common stock certain circumstances involving change control company company said conditions occur warrants would exercisable price equal pct common stock market price time exceed dlrs per share computer terminal also said sold rights dot impact technology including future improvements inc houston tex dlrs said would continue exclusive worldwide licensee technology company said moves part reorganization plan would help pay current operation costs ensure product delivery computer terminal makes computer generated labels forms ticket printers terminals reuter.\n###Output:\nacq\n\
\nGood example:\n\
###Input:\nmagma lowers copper cent cts magma copper co subsidiary newmont mining corp said cutting copper cathode price cent cents lb effective immediately reuter.\n###Output:\ncopper\n\
\nBad example:\n\
###Input:\nmagma lowers copper cent cts magma copper co subsidiary newmont mining corp said cutting copper cathode price cent cents lb effective immediately reuter.\n###Output:\nloss\n"

#     instruction = f"You are a text classifier and your task is to classifiy a given text into the following categories: "+ohsumed_classes +". \
# If you cannot decide, just output None. You should just output the label from the above categories. Do not output a sentence.\n\
# The meaning of each category is as follows: {'C01': 'Bacterial Infections and Mycoses', 'C02': 'Virus Diseases', 'C03': 'Parasitic Diseases', 'C04': 'Neoplasms', 'C05': 'Musculoskeletal Diseases', 'C06': 'Digestive System Diseases', 'C07': 'Stomatognathic Diseases', 'C08': 'Respiratory Tract Diseases', 'C09': 'Otorhinolaryngologic Diseases', 'C10': 'Nervous System Diseases', 'C11': 'Eye Diseases', 'C12': 'Urologic and Male Genital Diseases', 'C13': 'Female Genital Diseases and Pregnancy Complications', 'C14': 'Cardiovascular Diseases', 'C15': 'Hemic and Lymphatic Diseases', 'C16': 'Neonatal Diseases and Abnormalities', 'C17': 'Skin and Connective Tissue Diseases', 'C18': 'Nutritional and Metabolic Diseases', 'C19': 'Endocrine Diseases', 'C20': 'Immunologic Diseases', 'C21': 'Disorders of Environmental Origin', 'C22': 'Animal Diseases', 'C23': 'Pathological Conditions, Signs and Symptoms'}.\n\
# \nGood example:\n\
# ###Input:\ninfection total joint replacement although small number infections total joint replacements blood borne distant sources , infections appear derived operation strenuous attempts reduce risk cleaning air wound environment , coupled prophylactic antibiotics , reduced infection rates order magnitude decade time potential exchange arthroplasty established infection shown , results encouraging rigorous infection control key containing difficult expensive problem.\n###Output:\nC01\n\
# \nGood example:\n\
# ###Input:\nassessment platelet antibody flow cytometric elisa techniques comparison study two different methods evaluating platelet antibody used study 12 normal subjects 24 patients consisting primarily intravenous drug users \( ivdus \) positive human immunodeficiency virus \( hiv \) total platelet associated immunoglobulin g \( igg \) immunoglobulin \( igm \) measured enzyme lined immunosorbent assay platelet lysate , platelet surface associated igg igm measured semiquantitative flow cytometry igg igm values showed significant correlations two measurement methods mean platelet surface igg total igg 3 6 4 3 times greater , respectively , ivdus controls , platelet igm also significantly higher ivdus controls measured techniques although mean platelet immunoglobulin levels higher ivdus thrombocytopenia ivdus normal platelet counts , differences achieve significance data show platelet igg igm levels increased associated hiv infection suggest increases confined patients thrombocytopenia herein described platelet surface antibody total platelet antibody measurements appear equally useful studying patient population specific details generating platelet associated immunofluorescence units discussed.\n###Output:\nC02\n\
# \nGood example:\n\
# ###Input:\nkeratitis two species describe case keratitis related soft contact lens wear patient presented 3 week history severe pain , radial stromal infiltrates subepithelial infiltrates epithelial defect cultured corneal biopsy specimen , contact lens lens case corneal biopsy culture grew well escherichia coli patient treated topical \( \) drops , neomycin b drops gentamicin drops gradual clinical improvement ensued.\n###Output:\nC03\n\
# \nGood example:\n\
# ###Input:\nangle lipoma angle rare lesions date , 18 patients reported , 17 adults second child described angle lipoma.\n###Output:\nC04\n\
# \nBad example:\n\
# ###Input:\nangle lipoma angle rare lesions date , 18 patients reported , 17 adults second child described angle lipoma.\n###Output:\nNeoplasms\n"

    model = Chatting.ChatGPT(KEY, instruction)

    preds = []
    with open("./bertgcn_data/corpus/20ng.clean.txt", "r") as f:
        texts = f.readlines()
    texts = texts[:7532]

    print(f"total length of texts: {len(texts)}")

    for batch in tqdm(model.batch_call(texts, batch_size=32)):
        pred = [b[1][0] for b in batch]
        print(pred)
        preds.extend(pred)

    with open("./bertgcn_data/corpus/20ng.clean.pred.txt", "w") as f:
        preds = "\n\n\n\n".join(preds)
        f.write(preds)
