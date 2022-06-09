import os
MAX_LEN = 512
SEP_TOKEN_ID = 102
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tqdm
import time
import json
import numpy as np
import random
import torch
import functools
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import jsonlines
from transformers import BertModel, BertForSequenceClassification, BertConfig
from tqdm import tqdm
from math import sqrt

label_dict = {'Film': 0, 'Lake': 1, 'Language': 2, 'Company': 3, 'Person': 4, 'VideoGame': 5, 'City': 6, 'Currency': 7, 'Mountain': 8, 'Scientist': 9, 'Bird': 10, 'Plant': 11, 'TelevisionShow': 12, 'Animal': 13, 'Country': 14, 'AdministrativeRegion': 15, 'Genre': 16, 'Newspaper': 17, 'Airport': 18, 'AcademicJournal': 19, 'PopulatedPlace': 20, 'Wrestler': 21, 'PoliticalParty': 22, 'Cricketer': 23, 'Eukaryote': 24, 'Saint': 25, 'Writer': 26, 'Museum': 27, 'BaseballPlayer': 28, 'EducationalInstitution': 29, 'GovernmentType': 30, 'SportsTeam': 31, 'ChristianBishop': 32, 'Settlement': 33, 'Royalty': 34, 'EthnicGroup': 35, 'University': 36, 'Diocese': 37, 'Province': 38, 'ClubMoss': 39, 'GridironFootballPlayer': 40, 'CollegeCoach': 41, 'Colour': 42, 'Comedian': 43, 'Manga': 44, 'Publisher': 45, 'Magazine': 46, 'ComicsCharacter': 47, 'ComicsCreator': 48, 'Airline': 49, 'ConcentrationCamp': 50, 'Congressman': 51, 'Region': 52, 'Conifer': 53, 'SoccerClub': 54, 'Criminal': 55, 'Crustacean': 56, 'Curler': 57, 'Cycad': 58, 'Cyclist': 59, 'DartsPlayer': 60, 'Disease': 61, 'Drug': 62, 'Economist': 63, 'Organisation': 64, 'Embryology': 65, 'Engineer': 66, 'Entomologist': 67, 'Mammal': 68, 'Mollusca': 69, 'FashionDesigner': 70, 'Fern': 71, 'FictionalCharacter': 72, 'FigureSkater': 73, 'Wikidata:Q11424': 74, 'MusicalArtist': 75, 'Album': 76, 'Single': 77, 'FilmFestival': 78, 'Fish': 79, 'Food': 80, 'RacingDriver': 81, 'FormulaOneRacer': 82, 'GrandPrix': 83, 'Fungus': 84, 'GaelicGamesPlayer': 85, 'Town': 86, 'Village': 87, 'MusicGenre': 88, 'GolfPlayer': 89, 'GovernmentAgency': 90, 'Governor': 91, 'OfficeHolder': 92, 'FormulaOneTeam': 93, 'AmericanFootballPlayer': 94, 'AmericanFootballTeam': 95, 'Band': 96, 'Guitarist': 97, 'Instrument': 98, 'Gymnast': 99, 'AdultActor': 100, 'AmateurBoxer': 101, 'Ambassador': 102, 'Amphibian': 103, 'AnatomicalStructure': 104, 'Artery': 105, 'Muscle': 106, 'Nerve': 107, 'RaceHorse': 108, 'Anime': 109, 'Arachnid': 110, 'BritishRoyalty': 111, 'Architect': 112, 'Building': 113, 'Vein': 114, 'ArtificialSatellite': 115, 'School': 116, 'Astronaut': 117, 'Athlete': 118, 'AustralianRulesFootballPlayer': 119, 'Ship': 120, 'AutomobileEngine': 121, 'BadmintonPlayer': 122, 'Bank': 123, 'Baronet': 124, 'BaseballTeam': 125, 'BaseballLeague': 126, 'BasketballPlayer': 127, 'BasketballTeam': 128, 'BeachVolleyballPlayer': 129, 'BeautyQueen': 130, 'Protein': 131, 'Bodybuilder': 132, 'River': 133, 'Bone': 134, 'Book': 135, 'Artist': 136, 'Boxer': 137, 'Brain': 138, 'Bridge': 139, 'TelevisionStation': 140, 'Infrastructure': 141, 'BroadcastNetwork': 142, 'BusinessPerson': 143, 'Canal': 144, 'Canoeist': 145, 'Cardinal': 146, 'Castle': 147, 'Planet': 148, 'Chancellor': 149, 'Chef': 150, 'ChessPlayer': 151, 'ClassicalMusicArtist': 152, 'HandballPlayer': 153, 'Historian': 154, 'HockeyTeam': 155, 'IceHockeyPlayer': 156, 'HollywoodCartoon': 157, 'HorseRider': 158, 'HorseTrainer': 159, 'Hospital': 160, 'Hotel': 161, 'Insect': 162, 'Jockey': 163, 'Journalist': 164, 'Judge': 165, 'LacrossePlayer': 166, 'LawFirm': 167, 'Legislature': 168, 'Library': 169, 'Ligament': 170, 'Island': 171, 'Lymph': 172, 'MartialArtist': 173, 'Mayor': 174, 'Automobile': 175, 'Medician': 176, 'MemberOfParliament': 177, 'MilitaryConflict': 178, 'MilitaryPerson': 179, 'MilitaryUnit': 180, 'MilitaryStructure': 181, 'Aircraft': 182, 'PersonFunction': 183, 'Model': 184, 'Monarch': 185, 'Moss': 186, 'MotorcycleRider': 187, 'SpeedwayRider': 188, 'MountainRange': 189, 'MountainPass': 190, 'Murderer': 191, 'Song': 192, 'Musical': 193, 'NascarDriver': 194, 'NationalCollegiateAthleticAssociationAthlete': 195, 'NetballPlayer': 196, 'Noble': 197, 'Non-ProfitOrganisation': 198, 'OlympicEvent': 199, 'Swimmer': 200, 'OlympicResult': 201, 'TennisPlayer': 202, 'Painter': 203, 'Park': 204, 'Philosopher': 205, 'Photographer': 206, 'Play': 207, 'PlayboyPlaymate': 208, 'Poem': 209, 'Poet': 210, 'PokerPlayer': 211, 'Ideology': 212, 'President': 213, 'Politician': 214, 'Pope': 215, 'PowerStation': 216, 'RadioHost': 217, 'Place': 218, 'PrimeMinister': 219, 'Prison': 220, 'ProtectedArea': 221, 'PublicTransitSystem': 222, 'RadioProgram': 223, 'RadioStation': 224, 'RailwayLine': 225, 'Station': 226, 'RouteOfTransportation': 227, 'RailwayStation': 228, 'Tunnel': 229, 'RailwayTunnel': 230, 'Religious': 231, 'Reptile': 232, 'Restaurant': 233, 'RoadTunnel': 234, 'Rocket': 235, 'Rower': 236, 'HistoricBuilding': 237, 'ReligiousBuilding': 238, 'RugbyPlayer': 239, 'RugbyClub': 240, 'ScreenWriter': 241, 'Sea': 242, 'Senator': 243, 'ShoppingMall': 244, 'SiteOfSpecialScientificInterest': 245, 'Skater': 246, 'SkiArea': 247, 'Skier': 248, 'SnookerChamp': 249, 'SnookerPlayer': 250, 'SoccerLeague': 251, 'SoccerManager': 252, 'SoccerPlayer': 253, 'Software': 254, 'TennisTournament': 255, 'SquashPlayer': 256, 'SumoWrestler': 257, 'TableTennisPlayer': 258, 'TelevisionEpisode': 259, 'TelevisionHost': 260, 'Theatre': 261, 'Valley': 262, 'VoiceActor': 263, 'Volcano': 264, 'VolleyballCoach': 265, 'VolleyballPlayer': 266, 'Weapon': 267, 'WineRegion': 268, 'Grape': 269, 'WrestlingEvent': 270, 'YearInSpaceflight': 271, 'Cleric': 272, 'CyclingTeam': 273, 'HistoricPlace': 274}

def load_jsonl(jsonl_path, label_dict):
    target_cols = []
    labels = []
    fd_cols = []
    headers_alias = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    mapping = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25}
    with open(jsonl_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            target_cols.append(np.array(item['content'])[:,int(item['target'])])
            target_alias = headers_alias[int(item['target'])]
            fds = item['FD']
            fd_col_dict = {}
            fd_col = []
            max_size = int(sqrt(len(item['header'])))
            for fd_pair in fds:
                if fd_pair[1] == target_alias:
                    for char in fd_pair[0]:
                        if not mapping[char] in fd_col_dict.keys():
                            fd_col_dict[mapping[char]] = 1
                        else:
                            fd_col_dict[mapping[char]] += 1
                if fd_pair[0] == target_alias:
                    for char in fd_pair[1]:
                        if not mapping[char] in fd_col_dict.keys():
                            fd_col_dict[mapping[char]] = 1
                        else:
                            fd_col_dict[mapping[char]] += 1
            fd_sorted_dict = dict(sorted(fd_col_dict.items(), key=lambda x:x[1], reverse=True))
            for i in range(min(max_size,len(fd_sorted_dict))):
                fd_col.append(np.array(item['content'])[:,int(list(fd_sorted_dict.keys())[i])])
            labels.append(int(label_dict[item['label']]))
            fd_cols.append(fd_col)
    return target_cols, fd_cols, labels

class TableDataset(Dataset):
    def __init__(self, target_cols, tokenizer, fd_cols, labels):
        self.labels = labels
        self.target_cols = target_cols
        self.tokenizer = tokenizer
        self.fd_cols = fd_cols

    def tokenize(self, col):
        text = ''
        for cell in col:
            text+=cell
            text+=' '
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __getitem__(self, idx):
        target_token_ids = self.tokenize(self.target_cols[idx])
        fd_token_ids = []
        if len(self.fd_cols[idx]) == 0:
            fd_token_ids.append(target_token_ids)
        else:
            for col in self.fd_cols[idx]:
                fd_token_id= self.tokenize(col)
                fd_token_ids.append(fd_token_id)
        fd_token_ids = torch.stack(fd_token_ids)
        return target_token_ids, fd_token_ids, torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        fd_ids = [x[1] for x in batch]
        labels = torch.stack([x[2] for x in batch])
        return token_ids, fd_ids, labels

def get_loader(target_cols, fds, labels,batch_size=8,is_train=True):
    ds_df = TableDataset(target_cols, Tokenizer, fds, labels)
    loader = torch.utils.data.DataLoader(ds_df, batch_size=batch_size, shuffle=is_train, num_workers=0, collate_fn=ds_df.collate_fn)
    loader.num = len(ds_df)
    return loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class FEDORA(torch.nn.Module):
    def __init__(self, n_classes=275, dim_k=768, dim_v=768, num_heads=8):
        super(FEDORA, self).__init__()
        self.model_name = 'FEDORA'
        self.num_heads = num_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1/sqrt(dim_k//num_heads)
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.fd_bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.fcc= torch.nn.Linear(768, n_classes)
        self.linear_q = torch.nn.Linear(768, dim_k, bias = False)
        self.linear_k = torch.nn.Linear(768, dim_k, bias = False)
        self.linear_v = torch.nn.Linear(768, dim_v, bias = False)

    def attention(self, target_col, fds):
        cols = []
        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v //nh
        for ids in fds:
            attention_mask = (ids > 0)
            _, cur_pooled = self.fd_bert_model(input_ids=ids, attention_mask=attention_mask, return_dict=False)
            cols.append(cur_pooled)
        target_col = target_col.reshape(1,-1)
        fd_cols = torch.stack(cols).view(-1,768)
        concated = torch.cat((target_col, fd_cols)).reshape(1,-1,768)
        q = self.linear_q(concated).reshape(1,-1,nh,dk).transpose(1,2)
        k = self.linear_k(concated).reshape(1,-1,nh,dk).transpose(1,2)
        v = self.linear_v(concated).reshape(1,-1,nh,dv).transpose(1,2)
        dist = torch.matmul(q,k.transpose(2,3))*self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        attention_out = torch.matmul(dist,v)
        attention_out = attention_out.transpose(1,2).reshape(1,-1,self.dim_v)
        attention_out = attention_out[:,0,:]
        return attention_out

    def forward(self,ids,fds):
        attention_mask = (ids > 0)
        _, pooled =self.bert_model(input_ids=ids, attention_mask=attention_mask, return_dict=False)
        attention_out = self.attention(pooled, fds)
        out=self.dropout(attention_out)
        out = self.fcc(out)
        out = out.view(-1, 275)
        
        return out

def metric_fn(preds, labels):
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }


def train_model(model,train_loader,val_loader,lr,model_save_path='pytorch_FEDORA_model.pkl',early_stop_epochs=5,epochs=20):  
    no_improve_epochs = 0
    accumulation_steps = 1
    weight_decay = 1e-2
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    cur_best_v_loss =10.0
    for epoch in range(1,epochs+1):
        model.train()
        bar1 = tqdm(train_loader)
        epoch_loss = 0
        epoch_acc = 0
        v_epoch_loss = 0
        v_epoch_acc = 0
        for i,(ids, fds, labels) in enumerate(bar1):
            labels = labels.cuda()
            fds = [fd.cuda() for fd in fds]
            output = model(ids.cuda(), fds)
            y_pred_prob = output
            y_pred_label = y_pred_prob.argmax(dim=1)
            loss = loss_fn(y_pred_prob.view(-1, 275), labels.view(-1))/accumulation_steps
            acc = ((y_pred_label == labels.view(-1)).sum()).item()
            loss.backward()
            if ((i+1)%accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()*accumulation_steps
            epoch_acc += acc
            length_label = len(labels)
            del ids, fds, labels
            torch.cuda.empty_cache()
        train_length = len(bar1)+1
        print("Epoch:", epoch, "training_loss:", epoch_loss / (train_length), "\t", "current acc:", epoch_acc / ((train_length)-1+length_label))
        model.eval()
        bar2 = tqdm(val_loader)
        pred_labels = []
        true_labels = []
        for j,(ids, fds, labels) in enumerate(bar2):
            labels = labels.cuda()
            fds = [fd.cuda() for fd in fds]
            output = model(ids.cuda(), fds)
            y_pred_prob = output
            y_pred_label = y_pred_prob.argmax(dim=1)
            vloss = loss_fn(y_pred_prob.view(-1, 275), labels.view(-1))
            acc = ((y_pred_label == labels.view(-1)).sum()).item()
            pred_labels.append(y_pred_label.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())
            v_epoch_loss += vloss.item()
            v_epoch_acc += acc
            v_length_label = len(labels)
            del ids, fds
            torch.cuda.empty_cache()
        pred_labels = np.concatenate(pred_labels)
        true_labels = np.concatenate(true_labels)
        val_length = len(bar2)+1
        print("validation_loss:", v_epoch_loss / (val_length), "\t", "current acc:", v_epoch_acc / (val_length-1+v_length_label))
        f1_scores = metric_fn(pred_labels, true_labels)
        print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
        if v_epoch_loss / (val_length) < cur_best_v_loss:
            torch.save(model.state_dict(),model_save_path)
            cur_best_v_loss = v_epoch_loss / (val_length)
            no_improve_epochs = 0
            print("model updated")
        else:
            no_improve_epochs += 1
        if no_improve_epochs == 5:
            print("early stop!")
            break

def test_model(model,test_loader,lr,model_save_path='pytorch_FEDORA_model.pkl',early_stop_epochs=5,epochs=20):  
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    bar = tqdm(test_loader)
    pred_labels = []
    true_labels = []
    for i, (ids, fds, labels) in enumerate(bar):
        labels = labels.cuda()
        fds = [fd.cuda() for fd in fds]
        output = model(ids.cuda(), fds)
        y_pred_prob = output
        y_pred_label = y_pred_prob.argmax(dim=1)
        pred_labels.append(y_pred_label.detach().cpu().numpy())
        true_labels.append(labels.detach().cpu().numpy())
        del ids, fds
        torch.cuda.empty_cache()
    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    f1_scores = metric_fn(pred_labels, true_labels)
    print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
    return f1_scores['weighted_f1'], f1_scores['macro_f1']

if __name__ == '__main__':
    setup_seed(20)
    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    jsonl_path = "./data/train_val.jsonl"
    target_cols, fds, labels = load_jsonl(jsonl_path, label_dict)
    sfolder_cv = StratifiedKFold(n_splits=5, random_state = 0, shuffle=True)
    BS = 1
    lrs = [1e-5]
    for lr in lrs:
        print("start for learning rate:", lr)
        for cur_fold, (train_idx, val_idx) in enumerate(sfolder_cv.split(target_cols, labels)):
            ## if cur_fold == 0: #(this line is for parallel training the model on five folds, uncomment this line and adjust the indentation properly, you can then select the fold to be trained by changing 0 to 1,2,3,4)
            train_cols = []
            train_fds = []
            train_labels = []
            val_cols = []
            val_fds = []
            val_labels = []
            for t_idx in train_idx:
                train_cols.append(target_cols[t_idx])
                train_fds.append(fds[t_idx])
                train_labels.append(labels[t_idx])
            for v_idx in val_idx:
                val_cols.append(target_cols[v_idx])
                val_fds.append(fds[v_idx])
                val_labels.append(labels[v_idx])  
            train_loader = get_loader(train_cols, train_fds, train_labels, BS, True)
            val_loader = get_loader(val_cols, val_fds, val_labels, 1, False)
            model = FEDORA().cuda()
            model_save_path = './checkpoints/FEDORA_FD'+"_lr="+str(lr)+'_{}.pkl'.format(cur_fold+1)
            print("Starting fold", cur_fold+1)
            train_model(model, train_loader, val_loader,lr, model_save_path=model_save_path)
    
    print("###############################") # Testing code, comment it if you want to train and test seperately.
    test_jsonl_path = "./data/test.jsonl"
    test_target_cols, test_fds, test_labels = load_jsonl(test_jsonl_path, label_dict)
    test_loader = get_loader(test_target_cols, test_fds, test_labels, 1, False)
    
    for lr in lrs:
        print("start for testing learning rate:", lr)
        weighted_f1s = []
        macro_f1s = []
        for cur_fold in range(5):
            model = FEDORA().cuda()
            model_save_path = './checkpoints/FEDORA_FD'+"_lr="+str(lr)+'_{}.pkl'.format(cur_fold+1)
            print("Starting fold", cur_fold+1)
            cur_w, cur_m = test_model(model, test_loader,lr, model_save_path=model_save_path)
            weighted_f1s.append(cur_w)
            macro_f1s.append(cur_m)
        print("The mean F1 score is:", np.mean(weighted_f1s))
        print("The sd is:", np.std(weighted_f1s))
        print("The mean macro F1 score is:", np.mean(macro_f1s))
        print("The sd is:", np.std(macro_f1s))
        print("===============================")
