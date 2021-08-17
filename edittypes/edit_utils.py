from .wiki_edit_util import *
from numpy.linalg import norm
import mwparserfromhell
import numpy as np
from numpy import dot
from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer
from .datasources.diff import operations_with_context, _get_context_before,\
_get_context_after, para_operations_with_context
from revscoring.languages.english import safe_dictionary_check
from mwtext.content_transformers import Wikitext2Words
import nltk
from nltk.corpus import stopwords
from mwapi import Session
from revscoring.extractors import api
from revscoring.dependencies import solve
from nltk.metrics.distance import edit_distance

# TODO: Exclude abbreviations
#SENT_SPLIT_RE = '\. |\.\n|\.\r\n|\n|\r\n'
# https://regex101.com/r/nG1gU7/27
SENT_SPLIT_RE = '(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|\n|\r\n'

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
forbidden_link_prefixes = ['category', 'image', 'file']
wtpp = Wikitext2Words(forbidden_link_prefixes)
exclusions = [r'<!--.*?-->', r'{{.*?}}', r'&[a-z]+;',]
strip_regex = re.compile('|'.join(exclusions))
cite_tags = ['{{cite', '<ref']

def has_valid_words(words):
    is_valid = False
    for word in words:
        if safe_dictionary_check(word):
            is_valid = True
            break
    return is_valid

def num_proper_nouns(sentence):
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    np = 0
    for tag in tags:
        if tag[1] == 'NNP':
            np += 1
            
    return np

# https://github.com/mediawiki-utilities/python-mwtext/blob/master/mwtext/content_transformers/wikitext2words.py
def clean_refs(text):
    ref_re = "<ref[^<>]*>[^<]*<\/ref>"
    # Clean references without closing </ref> tag
    text = re.sub(r'<ref[^>]+/>', '', text)
    # Clean references with a closing </ref> tag
    text = re.sub(r'<ref.*?</ref>', '', text, flags = re.DOTALL)
    # Clean again for unclosed references
    text = re.sub('<ref[^<>]*>[^<]*','', text)
    return text

def remove_stopwords(inserted):
    return [w for w in inserted if not w in stop_words]

def clean_wikitext(text):
    text = clean_refs(text)
    return ' '.join(wtpp.transform(text))

def preprocess_wikitext(text):
    text = clean_refs(text)
    text = clean_wikitext(text)
    return remove_stopwords(text.split())

def last_open(text, open_tok, close_tok):
    is_last_open = False
    pos = text.rfind(open_tok)
    position = 0
    if pos != -1:
        position = pos
        if close_tok not in text[pos:]:
            is_last_open = True
    return is_last_open

def first_close(text, open_tok, close_tok):
    is_first_open = False
    pos = text.find(close_tok)
    position = 0
    if pos != -1:
        position = len(text)
        if open_tok not in text[:pos]:
            is_first_open = True
    return is_first_open

def has_unclosed_brace(opr, data, open_close_tokens):
    context_before, deleted, inserted, context_after = get_data_from_operations(opr, data)
    open_tok, close_tok = open_close_tokens
    is_open = False
    # context_before
    # Check if the last open brace in before context has a closing brace
    cb_last_open = last_open(context_before, open_tok, close_tok)
    if cb_last_open:
        is_open = True
    # inserted/delete
    # Check if the last open brace has a closing brace
    is_last_open = last_open(inserted, open_tok, close_tok)
    # Check if the first closing brace has an open brace
    is_first_close = first_close(inserted, open_tok, close_tok)
    if is_last_open or is_first_close:
        is_open = True
    # context_after
    # Check if the first closing brace in after context has an open brace
    ca_first_close = first_close(context_after, open_tok, close_tok)
    if ca_first_close:
        is_open = True
    if cb_last_open and is_last_open and is_first_close and ca_first_close:
        is_open = False
    return is_open


def ratio_matching_changes(opr, data):
    #is_matching = False
    if opr != 'replace':
        return 0#is_matching
    context_before, deleted, inserted, context_after = get_data_from_operations(opr, data)
    deleted = clean_refs(deleted)
    inserted = clean_refs(inserted)

    forbidden_link_prefixes = ['category', 'image', 'file']
    wtpp = Wikitext2Words(forbidden_link_prefixes)
    deleted = wtpp.transform(deleted)
    inserted = wtpp.transform(inserted)
    #deleted = re.sub('[^a-zA-Z0-9 ]', '', deleted)
    #inserted = re.sub('[^a-zA-Z0-9 ]', '', inserted)
    if len(inserted) == 0 and len(deleted) == 0:
        return 0#is_matching
    matching_changes = stem_overlap(inserted, deleted)
    return float(matching_changes / max(len(inserted), len(deleted)))
    # is_matching = float(matching_changes / max(len(inserted), len(deleted))) > 0.4
    # return is_matching


def get_paragraphs_for_revision(revid):
    session = Session("https://en.wikipedia.org/w/api.php", user_agent="test")
    api_extractor = api.Extractor(session)
    ds = [para_operations_with_context]
    datavalue = list(api_extractor.extract(revid, ds))
    paragraphs = get_observations_by_paragraph(datavalue[0])
    return paragraphs

def get_data_from_operations(opr, data):
    '''
    Given a operation type and associated data, returns the context_before,
    deleted, inserted, and context_after
    '''
    context_before = data[0][1]
    context_after = data[-1][1]
    deleted, inserted = '', ''
    if opr == "insert":
        inserted = data[1][1]
    elif opr == "replace":
        # code for replace segments
        deleted = data[1][1]
        inserted = data[2][1]
    elif opr == "delete":
        # code for deletion segments
        deleted = data[1][1]
    return context_before, deleted, inserted, context_after

def get_observations_by_paragraph(operations_with_para_context):
    # returns list of lists of operations in the same paragraph
    paragraphs = []
    for obs in operations_with_para_context:
        opr = obs[0]
        data = obs[1]
        context_before = data[0][1]
        if opr == 'equal':
            continue
        # add a new list for paragraph 
        if not paragraphs:
            paragraphs.append([])
        else:
            # check if not same paragraph
            last_obs = paragraphs[-1][-1]
            last_data = last_obs[1]
            # Use last_context_before as a default to compare against
            # context_before
            last_context_before = last_data[0][1]
            # If last_context_before is empty, we need to compare against the
            # deleted segment as the starting of the paragraph is deleted.
            if last_context_before == '' and \
                    (last_obs[0] == 'deleted' or last_obs[0] == 'replace'):
                last_context_before = last_data[1][1]
            end = min(len(context_before), len(last_context_before))
            
            # If the context_before is empty, its a new paragraphs
            if (len(context_before) == 0 ) :
                paragraphs.append([])
            # if their contexts are not the same, they are in different paragraphs
            elif context_before[:end] != last_context_before[:end]:
                paragraphs.append([])
            
        # append current observation
        paragraphs[-1].append(obs)
    return paragraphs

def get_paragraph(paragraph_edits, original = True, upto = None):
    '''
    Given a set of edits in a paragraph, returns the complete/partial paragraph before
    or after the edit
    upto: (opr, inserted, deleted) segment upto which paragraph to return, if None,
    returns full paragraph. Should be consistent with argument `original`
    '''
    curr_idx = 0
    para_so_far = ''
    last_context_after = ''
    for opr, data in paragraph_edits:
        context_before, deleted, inserted, context_after = get_data_from_operations(opr, data)
        compare_against = upto
        if len(para_so_far) == 0:
            # Add full context if the first segment
            para_so_far += context_before
        else:
            # Add only new context
            para_so_far += context_before[curr_idx:]
        if original:
            para_so_far += deleted
        else:
            para_so_far += inserted
        if upto and upto[0] == opr and upto[1] == deleted and upto[2] ==\
        inserted:
            return para_so_far
        # Update the index in context_before till which context accounted for
        curr_idx = len(context_before) + len(deleted)
        last_context_after = context_after
    return para_so_far + last_context_after

def get_changes_for_chunk(paragraph, opr, data, parse = True):
    sentence_original = get_containing_sentence(paragraph, opr, data).strip('.  \n')
    sentence_modified = get_containing_sentence(paragraph, opr, data, False).strip('. \n')
    lim = 5
    # Return False if completely different sentences extracted
    #if sentence_original[:lim] != sentence_modified[:lim] and\
    #sentence_original[-lim:] != sentence_modified[-lim:] and\
    #len(sentence_original.split()) > 10:
    #    return False
    if parse:
        sentence_original = mwparserfromhell.parse(sentence_original).strip_code()
        sentence_modified = mwparserfromhell.parse(sentence_modified).strip_code()
    words_removed, words_inserted = get_changes_for_sentence(sentence_original,
            sentence_modified)
    return (sentence_original, sentence_modified, words_removed, words_inserted)

def get_changes_for_sentence(sentence_original, sentence_modified):
    '''
    Take in an (opr, data) segment and get all deleted, inserted parts of the
    sentence containing that segment
    '''
    words_original = sentence_original.split()
    words_modified = sentence_modified.split()
    words_removed = [w for w in words_original if w not in words_modified]
    words_inserted = [w for w in words_modified if w not in words_original]
    return (words_removed, words_inserted)

def get_containing_sentence(para, opr, data, original = True):
    '''
    Given a paragraph and a segment with operation, returns the before and after
    versions of the containing sentence
    '''
    original_sentence, modified_sentence = '', ''
    sentence = ''
    # get data associated with the segment
    context_before, deleted, inserted, context_after = get_data_from_operations(opr, data)
    segment_upto = (opr, deleted, inserted)
    # clean before context
    # Get the context upto the deleted/inserted segment
    context_before = get_paragraph(para, original = original, upto = segment_upto)
    context_before = clean_refs(context_before)
    # get full paragraph and clean it
    full_para = get_paragraph(para, original = original)
    full_para = clean_refs(full_para)
    # get sentences
    para_sentences = re.split(SENT_SPLIT_RE, full_para)
    before_sentences = re.split(SENT_SPLIT_RE, context_before)
    #pdb.set_trace()
    # Construct partial sentence
    sentence_upto = before_sentences[-1]
    # Match the partially reconstructed sentence in the modified and 
    # original sentences to get the containing sentence
    sentences = []  
    for sent in para_sentences:
        if sentence_upto in sent:
            sentence = sent
            break
    return sentence

def get_segment_sentence(inserted, context_before, context_after):
    # reconstruct sentence
    sentence = ''
    
    # context before contains the begginning of the sentence
    if len(context_before) > 1 and context_before[-1] != '.' and context_before[-2] != '.':
        # get last sentence segment
        sentence = context_before.split('. ')[-1]
        
    split = inserted.split('. ')
    # inserted starts with a period, so ignore the previous segment
    if len(inserted) > 1 and (inserted[0] == '.' or inserted[1] == '.'):
        sentence = ''
    # inserted has the end of the sentence
    elif len(split) > 1:
        sentence = sentence + split[0]
        return sentence
        
    #sentence += ' '
    sentence += inserted
    if len(inserted) > 1 and (inserted[-1] == '.' or inserted[-2] == '.'):
        return sentence
    if len(context_after) > 1:
        #sentence += ' '
        sentence += context_after.split('. ')[0]
    return sentence

def get_limited_context(data):
    ''' Returns the context restricted to the same line '''
    context_before, context_after = data[0][1], data[-1][1]
    delim_before = context_before.rfind('\n')
    if delim_before == -1:
        delim_before = -1
    delim_after = context_after.find('\n')
    if delim_after == -1:
        delim_after = len(context_after)
    return context_before[delim_before + 1:], context_after[:delim_after]

def ambiguous_article_or_pronoun(seg_insert_stripped, seg_delete):
    words = ['he', 'she', 'they', 'them', 'her', 'him', 'his', 'it']
    if len(seg_insert_stripped) == 1:
        seg_insert = seg_insert_stripped[0]
    else:
        seg_insert = ""
        
    insert = False
    delete = False
    for word in words:
        if word in seg_insert.lower():
            insert = True
            
        if word == seg_delete.lower():
            delete = True

    # we want to delete and insert something else
    return delete and not insert

def is_significant(inserted, sentence):
    # returns true if inserted words are significant
    significant = ['ADV', 'ADJ', 'NOUN']
    
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens, tagset='universal')
    count = 0
    total = len(inserted)
    
    for tag in tags:
        if tag[0] in inserted and tag[1] in significant:
            count += 1

    if total == 0:
        return False
    proportion = float(count / total)
    return proportion > 4.0/5.0

def is_date(inserted):
    year = re.search(r'%d%d%d%d', inserted)
    if year:
        return True
    
    months = ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'November', 'December']
    for month in months:
        if re.search(month, inserted):
            return True
        
    return False

def seg_avg_len(segment):
    size = 0.0
    for line in segment:
        size += len(line)
    return size / max(1, len(segment))

def seg_min_len(segment):
    min_size = 100000.0
    for line in segment:
        min_size = min(len(segment), min_size)
    return min_size

def markup_chars_ratio_avg(segment):
    n_seg = len(segment)
    if n_seg == 0:
        return 0.0
    seg_ratio = []
    for line in segment:
        seglen = len(line)
        n_markups = segment_count(line, W_MARKUP)
        seg_ratio.append(n_markups / max(1, seglen))
    return 1.0 * sum(seg_ratio) / max(1, n_seg)

def markup_chars_ratio_min(segment):
    if len(segment) == 0:
        return 0.0
    seg_ratio = []
    for line in segment:
        seglen = len(line)
        n_markups = segment_count(line, W_MARKUP)
        seg_ratio.append(n_markups / max(1, seglen))

    return 1.0 * min(*seg_ratio, 1)

def markup_chars_ratio_max(segment):
    if len(segment) == 0:
        return 0.0
    seg_ratio = []
    for line in segment:
        seglen = len(line)
        n_markups = segment_count(line, W_MARKUP)
        seg_ratio.append(n_markups / max(1, seglen))

    return 1.0 * max(*seg_ratio, 0)


def seg_max_len(segment):
    max_size = 0.0
    for line in segment:
        max_size = max(max_size, len(segment))
    return max_size


def avg_words(words):
    num = 0
    vec = np.zeros(shape=(50,))
    for word in words:
        if word in enwiki_kvs:
            vec += enwiki_kvs[word]
            num += 1
    return vec / max(num, 1)

def elab_ratio(operations_with_context):
    max_sim = 0
    before = ''
    after = ''
    for opr, data in operations_with_context:
        if opr == 'insert':
            before = data[0][1] + data[2][1]
            after = data[0][1] + data[1][1] + data[2][1]
        elif opr == 'replace':
            before = data[0][1] + data[1][1] + data[3][1]
            after = data[0][1] + data[2][1] + data[3][1]
        before = mwparserfromhell.parse(before).strip_code().split()
        after = mwparserfromhell.parse(after).strip_code().split()
        diff = set(after) - set(before)

        beforevec = avg_words(before)
        aftervec = avg_words(after)
        sim = dot(beforevec, aftervec) / (norm(beforevec) * norm(aftervec))
        max_sim = max(sim, max_sim)
    return max_sim


def only_infobox(operations_with_context):
    infobox = False
    only_infobox = True
    for opr, data in operations_with_context:
        if opr == 'equal':
            continue
        ctx = data[0][1]
        if len(re.findall('\s*?[\|\*\!]', ctx)):
            infobox = True
        elif len(ctx) > 2:
            only_infobox = False
    return 1 if (only_infobox and infobox) else 0


def get_segment_cats(operations_with_context):
    classes = []
    owc_left = []
    classes.append(fact_update(operations_with_context))
    classes.append(refact(operations_with_context))
    classes.append(wikif(operations_with_context))
    classes.append(simplif(operations_with_context))
    classes.append(find_elab(operations_with_context))
    classes.append(verif(operations_with_context))
    classes.append(process(operations_with_context))
    classes.append(clarif(operations_with_context))
    classes.append(disamb(operations_with_context))
    classes.append(copyedit(operations_with_context))
    return classes


def grammar_cats(operations_with_context):
    for opr, data in operations_with_context:
        if opr == 'equal':
            continue
        sent = ''
        if opr == 'replace':
            sent = data[0][1] + data[1][1] + data[3][1]
        elif opr == 'insert' or opr == 'delete':
            sent = data[0][1] + data[1][1] + data[2][1]
        #pdb.set_trace()

def wikification_util(opr, data):
    markup_tokens = ['<sup', ' color', ' sub', ' px', ' align', '<br']
    present = 0
    seg = ''
    if opr == "insert":
        seg = data[1][1]
    elif opr == 'replace':
        seg = data[2][1]
    elif opr == 'delete':
        seg = data[1][1]
        if '[[Category' in data[1][1]:
            present = 1
        if '[' in seg:
            present = 1
    segments = seg.split('\n')
    context = data[0][1]
    for segment in segments:
        context_before = data[0][1]
        specials = re.findall(SPECIAL_TOK, segment)
        regs = re.findall(r'[a-zA-Z0-9]', segment)
        if len(specials) == 1 and specials[0] not in '[]':
            continue
        if 1.0 * len(specials) / max(len(regs), 1) > 0.5:
            present = 1
    if '[' in seg:
        present = 1
    for markup in MARKUP + markup_tokens:
        if markup in seg:
            present = 1 
    #pdb.set_trace()
    #if len(seg) == 0:
    #    present = 1
    #if len(re.findall(rbef, context)) > 0:
    #    present = 1

    #pdb.set_trace()
    return present


def factupdate_util(opr, data):
    fact_present = 0
    context = data[0][1]

    if verification_util(opr, data) == 1:
        return fact_present
    if '<ref' in context and '</ref' not in context:
        return fact_present
    if opr == 'replace':
        textwithoutspecials = re.sub('\(|\)|\"|\=|\[|\]|{|}|,|\||<|>', '',\
                data[2][1]).split()
        if len(textwithoutspecials) > 7:
            return fact_present
        if _detect_number_fact(data[2][1], data[0][1]) == 1:
            fact_present = 1
            #pdb.set_trace()
        if _detect_nums(data[2][1], data[0][1]) == 1:
            fact_present = 1
    return fact_present


def findelab(opr, data):
    elab = 0
    if opr == 'insert' and data[1][1].startswith('\n'):
        content = data[1][1].strip('\n')
        if len(re.findall('\\n', data[1][1])) > 3:
            curr_elab = 1
        if len(content) > 1 and not (content[0] in '{|['):
            # TODO: check presence of alphanumeric
            curr_elab = 1
    elif opr == 'insert' and len(re.findall('\\n', data[1][1])) > 2:
        curr_elab = 1
    elif opr == 'insert' and (len(data[0][1]) == 0 or len(data[2][1]) == 0):
        content = data[1][1].strip('\n')
        if len(content) > 1 and not (content[0] in '{|['):
            curr_elab = 1
    elif opr == 'replace' and len(re.findall('\\n', data[2][1])) > 2:
        # TODO: elimintate contiguous only \n, ending context should not be
        # considered
        curr_elab = 1
    if curr_elab == 1:
        elab = 1

    #pdb.set_trace()

    return elab


def processseg(opr, data):
    templates = ['cn', 'dead link', 'citation style', 'Citation needed', 
    'POV statement', 'cleanup-reorganize', 'sfn', 'Fact', 'clarify',
    'other people', 'deadurl', 'copy edit', 'uncategorized', 'Failed verification', 'Proposed deletion', 'Multiple issues', 'fact', 'citation needed', 'linkrot']
    temptext = ['{{' + temp for temp in templates]
    process = 0
    if opr == 'insert':
        text = data[1][1]
        for t in temptext:
            if t in text:
                process = 1
                #pdb.set_trace()
                break
    elif opr == 'replace':
        text = data[2][1]
        for t in temptext:
            if t in text:
                process = 1
                #pdb.set_trace()
                break
    return process


def disambiguation_util(opr, data):
    rbef = r'\[[^\]]+$'
    raft = r'\^[^\[]+\]'
    disamb = 0
    cbef = ''
    caft = ''
    delete = ''
    insert = ''
    if opr == 'insert' or opr == 'delete':
        cbef = data[0][1]
        caft = data[2][1]
        insert = data[1][1]
    elif opr == 'replace':
        cbef = data[0][1]
        delete = data[1][1]
        insert = data[2][1]
        caft = data[3][1]

    if '\n' in insert:
        return disamb
    if 'Category:' in cbef:
        disamb = 1
    if '[' in insert and ']' in insert:
        return disamb
    if ']' in insert and '|' in insert:
        disamb = 1
    if '[' in insert and '|' in insert:
        disamb = 1
    seg = cbef + delete + insert + caft
    if 'disambiguation' in seg or 'ambiguous' in seg:
        disamb = 1
    match = re.findall(rbef, cbef)
    if len(match) > 0:
        disamb = 1
    return disamb


def refactor(opr, data):
    refactored = 0
    phrasedel = set()
    phraseins = set()
    LIM = 10
    if opr == 'delete' or opr == 'replace':
        texts = data[1][1].strip().replace('.', '').split('\n')
        for text in texts:
            texttr = text[0:LIM]
            if len(texttr) < LIM:
                continue
            if texttr in phraseins and '\n' in data[1][1]:
                refactored = 1
                #pdb.set_trace()
                break
            phrasedel.add(texttr)
    elif opr == 'insert':
        texts = data[1][1].strip().replace('.', '').split('\n')
        for text in texts:
            texttr = text[0:LIM]
            if len(texttr) < LIM:
                continue
            if texttr in phrasedel and '\n' in data[1][1]:
                refactored = 1
                #pdb.set_trace()
                break
            phraseins.add(texttr)
    #pdb.set_trace()
    return refactored


def stem_check(worda, wordb):
    worda = worda.lower()
    wordb = wordb.lower()
    if len(worda) < len(wordb):
        worda, wordb = wordb, worda
    worda_st = ps.stem(worda)
    wordb_st = ps.stem(wordb)

    if worda.startswith(wordb):
        return 1
    if fuzz.ratio(worda, wordb) > 85:
        return 1
    return 0


def simplification_detection(operations_with_context):
    prev_cbef = ''
    prev_caft = ''
    CLIM = 20
    for opr, data in operations_with_context:
        if opr == 'equal':
            continue
        cbef = data[0][1][:20]
        caft = data[-1][1][-20:]
        if cbef == prev_cbef or caft == prev_caft:
            pdb.set_trace()
            # Still in the same para
        prev_cbef = cbef
        prev_caft = caft
        #if cbef == prev_cbef:



def copyedit(operations_with_context):
    copyed = 0
    same_seg_a = []
    same_seg_b = []
    LIM = 4
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        #copyed = copyedit_util(opr, data)
        context_before = data[0][1]
        context_after = ''
        segment = ''
        #pdb.set_trace()
        #if len(re.findall('[a-zA-Z0-9]', segment)) == 0:
        #    continue
        if opr == 'insert' or opr == 'delete':
            context_after = data[2][1]
            segment = data[1][1]
        elif opr == 'replace':
            context_after = data[3][1]
            segment = data[2][1]
        else:
            continue
        if len(segment) == 1 and len(re.findall('[a-zA-Z0-9]*', segment)) > 0:
            copyed = 1
        if len(segment) == 1 and len(re.findall('[ ,-\.\(\)\[\]\'\"]', segment)) > 0:
            copyed = 1

        if opr == 'replace' and len(segment.split()) == 1 and len(data[1][1].split()) == 1:
            stc = stem_check(data[1][1].split()[0], segment.split()[0])
            if stc == 1:
                copyed = 1

        before_short = " ".join(context_before.split()[0:LIM])
        after_short = " ".join(context_after.split()[-LIM:])
        added_b = 0
        added_a = 0
        for idx, seg in enumerate(same_seg_b):
            if seg[0] == before_short:
                same_seg_b[idx][1].append(segment)
                added_b = 1
        if added_b != 1:
            same_seg_b.append((before_short, [segment]))

        for idx, seg in enumerate(same_seg_a):
            if seg[0] == after_short:
                same_seg_a[idx][1].append(segment)
                added_a = 1
        if added_a != 1:
            same_seg_a.append((after_short, [segment]))
        #pdb.set_trace()
    same_seg_ct = 0
    for seg in same_seg_a + same_seg_b:
        if len(seg[1]) > 1:
            same_seg_ct += 1
    if same_seg_ct > 1:
        copyed = 1
    #pdb.set_trace()
    return copyed
          

def verification_util(opr, data):
    verif = 0
    seg = ''
    cbef = data[0][1]
    if opr == 'insert':
        seg = data[1][1]
    elif opr == 'replace':
        seg = data[2][1]
    seg = seg.lower()
    # Reference should not be present in inserted text
    if '<ref' in seg or '{{cite' in seg or 'http' in seg:
        verif = 1
        #pdb.set_trace()
    # If open cite or ref tags present in before context, its a reference
    # modification
    if '{{cite' in cbef and '}}' not in cbef:
        verif = 1
    if '<ref' in cbef and '</ref' not in cbef:
        verif = 1
    return verif

def is_only_verification(opr, data):
    is_only_verif = False
    context_before, delete, insert, context_after = get_data_from_operations(opr, data)
    if last_open(context_before, '{{cite', '}}') and first_close(context_after, '{{cite',
            '}}'):
        is_only_verif = 1
    if last_open(context_before, '<ref', '</ref') and first_close(context_after, '<ref',
            '</ref'):
        is_only_verif = 1
    return is_only_verif


segment_cats = Datasource("segment_cats",
            get_segment_cats,
            depends_on=[operations_with_context])



def get_brace_match(seg, cbef, caft, mode = 0):
    if len(seg) == 0:
        return 0
    if len(cbef) == 0:
        cbef = seg
    present = 0
    if '[' in seg and ']' not in seg:
        present = 1
    if len(re.findall(r'^\|\s*-$|^\|$', seg)) and cbef[0] == seg[0]:
        present = 1

    if mode == 1:
        return present
    # before after regex
    #bef = re.findall(r'\[[^\]]*?$', cbef)
    bef = re.findall(r'\=\s*$', cbef)
    if len(bef) > 0:
        present = 1
    return present


def wikif(operations_with_context):
    present = 0
    added = ''
    removed = ''
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        if opr == 'replace':
            removed = data[1][1]
            added += data[2][1]
        elif opr == 'delete':
            removed += data[1][1]
        elif opr == 'insert':
            added += data[1][1]
        present = wikification_util(opr, data)
        if present == 1:
            #pdb.set_trace()
            break
            #pdb.set_trace()
        #elif op == "delete":
        #    segment = data[1][1]
        #    context_before = data[0][1]
        #    context_after = data[2][1]
        #    if get_brace_match(segment, context_before, context_after, 1) == 1:
        #        present = 1
        #        break
    #for seg in all_segments:
    #    if get_brace_match(seg) == 1:
    #        present = 1
    #        break
    if len(added.split()) == 0 and len(removed.split()) == 0:
        present = 1
    return present


def _detect_number_fact(data, context):
    fact_present = 0
    #if '[' in data and ']' in data:
    #    return fact_present
    #pdb.set_trace()
    if len(re.findall('[a-zA-Z0-9]', data)) == 0:
        return fact_present
    bef = re.findall(r'\[[^\]]*?$', context)
    if len(bef) > 0 and ']' not in data and '|' not in data and '|' not in\
    context and 'ref' not in context and 'Category' not in context and 'cite'\
    not in context:
        #pdb.set_trace()
        #fact_present = 1
        abc=1
    infobox = re.findall(r'=\s*?$', context)
    if len(infobox) > 0 and 'http' not in data and '==' not in context:
        fact_present = 1
        #pdb.set_trace()
    return fact_present


def _detect_nums(data, context):
    nums = re.findall(r'[0-9]+?', data)
    num_words = re.findall(r'zero|one|two|three|four|five|six|seven|eight|nine|ten', data)
    specials = re.findall('\[|{|<|=|\/', data)
    tokens = wikitext_split.tokenize(data)
    #pdb.set_trace()
    if (len(nums) > 0 or len(num_words) > 0) and len(specials) == 0:
        #pdb.set_trace()
        return 1
    return 0

def fact_update(operations_with_context):
    fact_present = 0
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        context = data[0][1]

        if verification_util(opr, data) == 1:
            continue
        if '<ref' in context and '</ref' not in context:
            continue
        if opr == 'replace':
            textwithoutspecials = re.sub('\(|\)|\"|\=|\[|\]|{|}|,|\||<|>', '',\
                    data[2][1]).split()
            if len(textwithoutspecials) > 7:
                continue
            if _detect_number_fact(data[2][1], data[0][1]) == 1:
                fact_present = 1
                #pdb.set_trace()
            if _detect_nums(data[2][1], data[0][1]) == 1:
                fact_present = 1
                #pdb.set_trace()
        #pdb.set_trace()
        #elif opr == 'delete':
        #    if _detect_number_fact(data[1][1], data[0][1]) == 1:
        #        fact_present = 1
        #        #pdb.set_trace()
        #pdb.set_trace() 
    return fact_present


def find_elab(operations_with_context):
    elab = 0
    for chunk in operations_with_context:
        curr_elab = 0
        opr = chunk[0]
        data = chunk[1]
        content = ''
        if opr == 'insert' or opr == 'delete':
            content = data[1][1]
        elif opr == 'replace':
            content = data[2][1]
        if opr == 'insert' and data[1][1].startswith('\n'):
            content = data[1][1].strip('\n')
            if len(re.findall('\\n', data[1][1])) > 3:
                curr_elab = 1
            if len(content) > 1 and not (content[0] in '{|['):
                # TODO: check presence of alphanumeric
                curr_elab = 1
        elif opr == 'insert' and len(re.findall('\\n', data[1][1])) > 2:
            curr_elab = 1
        elif opr == 'insert' and (len(data[0][1]) == 0 or len(data[2][1]) == 0):
            content = data[1][1].strip('\n')
            if len(content) > 1 and not (content[0] in '{|['):
                curr_elab = 1
        elif opr == 'replace' and len(re.findall('\\n', data[2][1])) > 2:
            # TODO: elimintate contiguous only \n, ending context should not be
            # considered
            curr_elab = 1
        if curr_elab == 1:
            elab = 1
        wikif = wikification_util(opr, data)
        disamb = disambiguation_util(opr, data)
        verif = verification_util(opr, data)
        if wikif == 0 and disamb == 0 and verif == 0 and len(content.split()) >= 2:
            elab = 1
        #pdb.set_trace()
    return elab


def numbers_present(operations_with_context):
    max_ratio = 0.0
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        if opr == 'replace':
            nums = re.findall(r'([0-9]+)?', data[2][1])
            num_words = re.findall(r'zero|one|two|three|four|five|six|seven|eight|nine|ten',
                    data[2][1])
            num_chars = sum([len(k) for k in nums])
            num_chars += sum([len(k) for k in num_words])
            max_ratio = max(max_ratio, 1.0 * num_chars / max(1, len(data[2][1])))
    return max_ratio


def process(operations_with_context):
    templates = ['cn', 'dead link', 'citation style', 'Citation needed', 
    'POV statement', 'cleanup-reorganize', 'sfn', 'Fact', 'clarify',
    'other people', 'deadurl', 'copy edit', 'uncategorized', 'Failed verification', 'Proposed deletion', 'Multiple issues', 'fact', 'citation needed', 'linkrot']
    temptext = ['{{' + temp for temp in templates]
    process = 0
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        text = ''
        cbef = data[0][1]
        if opr == 'insert' or opr == 'delete':
            text = data[1][1].lower()
        elif opr == 'replace':
            text = data[2][1].lower()
        if len(re.findall(r'\n', text)) >= 3:
            continue
        if 'multiple issues' in text:
            process = 1
            break
        tpls = re.findall('{{[a-zA-Z]+', text)
        #pdb.set_trace()
        for tpl in tpls:
            if '{{cite' not in tpl.lower():
                process = 1
                break
        tpls = re.findall('{{[a-zA-Z]*', cbef)
        if '}}' in cbef:
            continue
        for tpl in tpls:
            if '{{cite' not in tpl.lower():
                process = 1
                break

        #pdb.set_trace()
    return process


def disamb(operations_with_context):
    disamb = 0
    rbef = r'\[[^\]]+$'
    raft = r'\^[^\[]+\]'
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        disamb = disambiguation_util(opr, data)
        if disamb == 1:
            break
    return disamb


def phrase_matcher(phrase, prevs):
    match = 0
    for prev in prevs:
        size = min(len(phrase), len(prev))
        match = 1.0 * len(phrase & prev) / size 
        if match != 0:
            break
    return match

def refact(operations_with_context):
    refactored = 0
    phrasedel = []
    phraseins = []
    LIM = 8
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        if opr == 'delete':
            texts = data[1][1].strip().replace('.', '').split('\n')
            for text in texts:
                texttr = set(text.split()[0:LIM])
                if len(texttr) < LIM:
                    continue
                if phrase_matcher(texttr, phraseins) > 0.5 and '\n' in data[1][1]:
                    refactored = 1
                    #pdb.set_trace()
                    break
                phrasedel.append(texttr)
        elif opr == 'replace':
            segdel = data[1][1]
            segadd = data[2][1]
            textsdel = segdel.strip().replace('.', '').split('\n')
            textsadd = segadd.strip().replace('.', '').split('\n')
            for text in textsdel:
                texttr = set(text.split()[0:LIM])
                if len(texttr) < LIM:
                    continue
                if phrase_matcher(texttr, phraseins) > 0.5 and '\n' in data[1][1]:
                    refactored = 1
                    #pdb.set_trace()
                    break
                phrasedel.append(texttr)
            for text in textsadd:
                texttr = set(text.split()[0:LIM])
                if len(texttr) < LIM:
                    continue
                if phrase_matcher(texttr, phrasedel) > 0.5 and '\n' in data[2][1]:
                    refactored = 1
                    #pdb.set_trace()
                    break
                phraseins.append(texttr)
        elif opr == 'insert':
            texts = data[1][1].strip().replace('.', '').split('\n')
            for text in texts:
                texttr = set(text.split()[0:LIM])
                if len(texttr) < LIM:
                    continue
                if phrase_matcher(texttr, phrasedel) > 0.5 and '\n' in data[1][1]:
                    refactored = 1
                    #pdb.set_trace()
                    break
                phraseins.append(texttr)
        #pdb.set_trace()
    return refactored


def clarif(operations_with_context):
    clari = 0
    rbef = r'\[[^\]]+$'
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        context = data[0][1]
        seg = ''
        if opr == 'insert':
            seg = data[1][1]
        elif opr == 'replace':
            seg = data[2][1]
        if len(re.findall(rbef, context)) > 0 and len(re.findall('[A-Z]', seg))\
        == 0:
            continue
        if '==' in seg or '==' in context:
            continue
        if disambiguation_util(opr, data) == 1 or factupdate_util(opr, data) ==\
        1 or verification_util(opr, data) == 1:
            continue
        #if len(re.findall('[A-Z]', seg)) == 0:
        #    continue
        clarif_specials = '\"|\=|\[|\]|{|}|\||<|>|;|#'
        seg = re.sub('\[\[[a-zA-Z0-9\ |]+\]\]', 'one', seg)
        specialchars = len(re.findall(clarif_specials, seg))
        regchars = len(re.findall('[a-zA-Z0-9]', seg))
        if specialchars / max(regchars, 1) > 0.4:
            continue
        segment = seg.split()
        segments = list(filter(lambda k: len(k) > 2, segment))
        #pdb.set_trace()
        if len(segments) < 8 and len(segments) > 0 and '\n' not in seg:
            #pdb.set_trace()
            clari = 1
            #break
    return clari


def simplification_util(opr, data):
    simpli = 0
    #pdb.set_trace()
    seg = ''
    if opr == 'equal':
        return simpli
    if '[' in data[0][1] and ']' not in data[0][1]:
        return simpli
    if len(data[1][1]) > 0 and 'Category:' in data[1][1]:
        return simpli
    #if len(data[0][1]) > 0 and data[0][1][0] == '|' and opr == 'replace':
    #    return simpli
    if wikification_util(opr, data) == 1 or disambiguation_util(opr, data)\
    == 1 or verification_util(opr, data) == 1:
        return simpli
    if opr == 'replace':
        removed = data[1][1]
        added = data[2][1]
        #if len(added) / max(len(removed), 1) > 0.5:
        #    return simpli
    if opr == 'delete' or opr == 'replace':
        seg = data[1][1]
        if len(re.findall('\n', seg)) >= 1:
            simpli = 1
        specials = '\(|\)|\"|\=|\[|\]|{|}|,|\||<|>|[0-9]'
        specialchars = len(re.findall(specials, seg))
        regchars = len(re.findall('[a-zA-Z]', seg))
        textwithoutspecials = re.sub(specials,
                '', seg).split()
        if 1.0 * regchars / max(specialchars, 1) > 0.3 and \
        len(textwithoutspecials) > 2:
            simpli = 1
        if opr == 'delete' and len(textwithoutspecials) >= 2:
            simpli = 1
        #if '=' in seg and 'http' not in seg:
        #    continue
        #segment = re.sub('[\[\]\{\}\(\)=]', ' ', data[1][1]).split()
        #segments = list(filter(lambda k: len(k) > 2, segment))
        #if len(re.findall('\n', seg)) >= 2 and len(segments) > 3:
        #    simpli = 1
        #    break
    return simpli



def simplif(operations_with_context):
    simpli = 0
    simplification_detection(operations_with_context)
    return 0
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        simpli = simplification_util(opr, data)
        if simpli == 1:
            break
    return simpli


def verif(operations_with_context):
    verif = 0
    for chunk in operations_with_context:
        opr = chunk[0]
        data = chunk[1]
        #pdb.set_trace()
        verif = verification_util(opr, data)
        if verif == 1:
            break
    return verif

def get_section_from_text(text):
    curr_section = ''
    sec_regex = r'\n==([^=]+)=='
    secs = re.findall(sec_regex, text)
    cs = ''
    for match in secs:
        cs = match
    return cs


def citation_util(operations_with_context):
    found = 0 
    before, after = [], []
    after_content = []
    curr_section = ''
    for opr, data in operations_with_context:
        pdb.set_trace()
        content = ''
        if opr == 'replace':
            content = data[2][1]
            before += data[0][1] + data[1][1] + data[3][1]
            after += data[0][1] + data[2][1] + data[3][1]
            after_content.append(data[2][1])
        elif opr == 'insert':
            content = data[1][1]
            before += data[0][1] + data[2][1]
            after += data[0][1] + data[1][1] + data[2][1]
            after_content.append(after)
            cs = get_section_from_text(data[1][1])
            if cs:
                curr_section = cs
        elif opr == 'equal':
            cs = get_section_from_text(data[0][1])
            if cs:
                curr_section = cs
            after_content.append(data[0][1])
        for tag in REFERENCE:
            if tag in content:
                found = 1
    pdb.set_trace()
    sections = mwparserfromhell.parse(after_content).get_sections()
    for section in sections:
        pdb.set_trace()
    return before, after, found

def citation_util_with_sections(diff_operations):
    ops, a, b = diff_operations

    i = 0
    last_section = '<>'
    citations_with_sections = []
    while i < len(ops):
        op = ops[i]

        if op.name == "delete":
            del_op = op
            if len(ops) > i + 1 and ops[i + 1].name == "insert":
                ins_op = ops[i + 1]
                ls = get_section_from_text(''.join(b[ins_op.b1:ins_op.b2]))
                if ls:
                    last_section = ls
                inserted = "".join(b[ins_op.b1:ins_op.b2])
                if '<ref' in inserted or 'cite' in inserted.lower():
                    inserted_with_section ="".join(_get_context_before(a,
                        op.a1)) + inserted + "".join(_get_context_after(a, op.a2)) 
                    citations_with_sections.append((inserted_with_section,
                        last_section))
                i += 1  # Increments past ins_op
            else:
                pass
        elif op.name == "insert":
            ls = get_section_from_text(''.join(b[op.b1:op.b2]))
            if ls:
                last_section = ls
            inserted = "".join(b[op.b1:op.b2])
            if '<ref' in inserted or 'cite' in inserted.lower():
                inserted_with_section ="".join(_get_context_before(a,
                    op.a1)) + inserted + "".join(_get_context_after(a, op.a2)) 
                citations_with_sections.append((inserted_with_section,
                    last_section))

        else:  # op.name == "equal"
            ls = get_section_from_text(''.join(b[op.b1:op.b2]))
            if ls:
                last_section = ls
        i += 1
    return citations_with_sections

def process_util(opr, data):
    label = 0
    context_before, deleted, inserted, context_after =\
    get_data_from_operations(opr, data)
    context_bef = data[0][1]
    inserted = ''
    context_aft = data[-1][1]
    if opr == "insert":
        inserted = data[1][1]
    elif opr == "replace":
        inserted = data[2][1]
    elif opr == "delete":
        # Treating the deleted segment same as inserted
        inserted = data[1][1]
    # If full tempalte insertion with very few extra words outside of it
    if '{{' in inserted and '}}' in inserted:
        # Remove text within template, so that we can count extra words
        temp_removed = re.sub('{{.*?}}', '', inserted)
        if len(temp_removed.split()) < 3:
            label = 1
    # Partial template modification
    # https://en.wikipedia.org/wiki/?diff=741819890
    # Tempalte over multiple lines
    # https://en.wikipedia.org/wiki/?diff=713769462
    # https://en.wikipedia.org/wiki/?diff=743516487
    partial_modification = re.findall('{{.*', inserted)
    if len(partial_modification) > 0:
        text = partial_modification[0]
        if '}}' not in text:
            label = 1

    # Internal template modification
    # https://en.wikipedia.org/wiki/?diff=745156189
    open_before = context_bef.rfind('{{')
    close_after = context_aft.find('{{')
    if open_before != -1 and close_after != -1:
        before = context_bef[open_before:]
        after = context_aft[:close_after]
        # If unmatched open brace in before context and unmatched close brace in after context,
        # its modification of template contents
        if '}}' not in before and '{{' not in after:
            label = 1
    #pdb.set_trace()
    return label

def detect_newlines_util(deleted, inserted):
    if '\n' in deleted or '\n' in inserted:
        return 1
    return 0

def small_edits(operations_with_context):
    edit_segments = 0
    label = 1
    for opr, data in operations_with_context:
        context_bef = data[0][1]
        context_aft = data[-1][1]
        deleted, inserted = '', ''
        if opr == 'insert':
            edit_segments += 1
            inserted = data[1][1]
        if opr == 'replace':
            deleted = data[1][1]
            inserted = data[2][1]
            edit_segments += 1
        if opr == 'delete':
            deleted = data[1][1]
            edit_segments += 1
        if detect_newlines_util(deleted, inserted):
            label = 0

    #if edit_segments > 3:
    #    label = 0
    return label


def is_clarification_segment(opr, data, debug = False):
    is_clarif = 0
    context_before, deleted, inserted, context_after = get_data_from_operations(opr, data)
    if process_util(opr, data):
        return is_clarif
    if is_only_verification(opr, data):
        return is_clarif
        
    # Delete reference tags
    inserted = clean_refs(inserted)
    # exclude templates
    # exclude wikilink modifications
    # exclude insertions like - "|Meyrick]]", still need to handle trailing spaces
    # exclude insertions like - "[[Edward", still need to handle leading spaces
    if has_unclosed_brace(opr, data, ('[[', ']]')):
        return is_clarif
    if has_unclosed_brace(opr, data, ('{{', '}}')):
        return is_clarif
    # exclude category
    if 'category:' in context_before.lower() or 'category:' in inserted.lower():
        return is_clarif
    # exclude infoboxes, lists
    if context_before.strip().startswith('|') or context_before.strip().startswith('*'):
        #if context_before.strip().endswith('='):
        #    return is_clarif
        return is_clarif

    if '==' in context_before or '==' in inserted or '==' in deleted:
        return is_clarif

    if detect_newlines_util(deleted, inserted):
        return is_clarif
    
    # Get rid of wiki-markup
    forbidden_link_prefixes = ['category', 'image', 'file']
    wtpp = Wikitext2Words(forbidden_link_prefixes)
    inserted_stripped = wtpp.transform(inserted)
    inserted_filtered = remove_stopwords(inserted_stripped)
    deleted_filtered = remove_stopwords(deleted.split())

    # clarification if few word additions, and limited word deletions
    # TODO: If its one word, it should better be a good word, not 'a', 'an', 'the'..., 
    # Also match with prev word to exclude spelling fixes
    sentence = get_segment_sentence(inserted, context_before, context_after)
    if len(inserted_filtered) > 0 and len(inserted_filtered) < 10 and\
    len(deleted_filtered) < 10 and len(inserted_filtered) - len(deleted_filtered) > 0:
        is_clarif = 1

    # TODO: Remove the below rules, these are taken care of by the above rule,
    # after stopwords removal
    #elif len(deleted_filtered) == 1 and len(inserted_stripped) < 5 and len(inserted_stripped) > 1 \
    #and ambiguous_article_or_pronoun(inserted _stripped, deleted.split()[0]):
    #    is_clarif = 1
    #      
    ##adding more specific words
    #elif len(inserted_filtered) > 2:
    #    if is_significant(inserted_stripped, sentence):
    #        is_clarif = 1
    #  
    #    if is_date(inserted) and len(inserted.split()) < 10:
    #        is_clarif = 1
            
    # TODO: Exclude heading changes
    # TODO: Handle refactored changes
    # TODO: Hanlde overlap with POV
    # TODO: To differentiate from copy-edit like edits - https://en.wikipedia.org/wiki/?diff=714378070
    # we will need to threshold on the number of segments changed in the same para
    # MAYBE: Think if we want to do multiple sentence additions - delimited by '.'
    if debug:
        pdb.set_trace()
    return is_clarif

def clarification_util(operations_with_context, debug = False):
    label = 0
    # labeling code goes here
    for opr, data in operations_with_context:
        label = is_clarification_segment(opr, data, debug)
        if label == 1:
            break
    return label

def clarification_util_para(paragraph_operations_with_context, debug = False):
    label = 0
    done = False
    paragraphs = get_observations_by_paragraph(paragraph_operations_with_context)
    for paragraph in paragraphs:
        # Skip too many changes
        #if len(paragraph) > 10:
        #    continue
        for opr, data in paragraph:
            cb, ca = get_limited_context(data)
            context_before, context_after = data[0][1], data[-1][1]
            data_new = None
            if opr == 'replace':
                data_new = (('context_before', cb), data[1], data[2], ('context_after', ca))
            else:
                data_new = (('context_before', cb), data[1], ('context_after', ca))
            #has_matching_changes(opr, data_new)
            if is_clarification_segment(opr, data_new, debug) == 1:
                label = 1
        if label == 1:
            break
    return label

def clarification_util_statements(paragraph_operations_with_context):
    paragraphs = get_observations_by_paragraph(paragraph_operations_with_context)
    #if len(paragraphs) > 10:
    #    return None
    clarified_statements = []
    old_statements_added = []
    span = 20
    for paragraph in paragraphs:
        # Skip too many changes
        if len(paragraph) > 7:
            continue
        for opr, data in paragraph:
            cb, ca = get_limited_context(data)
            context_before, context_after = data[0][1], data[-1][1]
            _, deleted, inserted, _ = get_data_from_operations(opr, data)
            if '.' in inserted:
                continue
            data_new = None
            if opr == 'replace':
                data_new = (('context_before', cb), data[1], data[2], ('context_after', ca))
            else:
                data_new = (('context_before', cb), data[1], ('context_after', ca))
            #has_matching_changes(opr, data_new)
            label = is_clarification_segment(opr, data_new, False)
            if label:
                old_sentence = get_containing_sentence(paragraph, opr, data, original
                        = True)
                if old_sentence[:span] in old_statements_added:
                    continue
                new_sentence = get_containing_sentence(paragraph, opr, data, original
                        = False)
                #pdb.set_trace()
                old_statements_added.append(old_sentence[:span])
                clarified_statements.append((old_sentence, new_sentence, inserted))
    return clarified_statements


def pov_deletions(para_operations_with_context):
    sentences = []
    old_statements_added = []
    SPAN = 15
    paragraphs = get_observations_by_paragraph(para_operations_with_context)
    if len(paragraphs) > 1:
        return sentences
    for paragraph in paragraphs:
        can_add = True
        to_add = []
        to_add_old = []
	# Skip large paragraphs
        if len(paragraph)  > 1:
            continue
        for opr, data in paragraph:
            data_new = None
            cb, ca = get_limited_context(data)
            _, deleted, inserted, _ = get_data_from_operations(opr, data)
            if opr == 'replace':
                data_new = (('context_before', cb), data[1], data[2], ('context_after', ca))
            else:
                data_new = (('context_before', cb), data[1], ('context_after', ca))
            data_old = data
            data = data_new
            
            # use has_unclosed_brace to exclude templates
            if has_unclosed_brace(opr, data, ('[[', ']]')):
                continue
            if has_unclosed_brace(opr, data, ('{{', '}}')):
                continue
            # use is_only_verification to exclude citations
            if is_only_verification(opr, data):
                continue
            if detect_newlines_util(deleted, inserted):
                continue
            if cb.strip().startswith('|') or cb.strip().startswith('*'):# or cb.strip().startswith('{'):
                continue
            if process_util(opr, data):
                continue
            if len(re.split(SENT_SPLIT_RE, deleted)) > 1:
                continue
            if len(re.split(SENT_SPLIT_RE, inserted)) > 1:
                continue
            sorig, smod, worig, wmod = get_changes_for_chunk(paragraph, opr,
                    data_old, parse = False)
            if '\n' in sorig or '\n' in deleted.strip('\n') or '\n' in\
                inserted.strip('\n') or '\n' in smod:
                continue
            # https://arxiv.org/pdf/1911.09709.pdf
            # maximal - more than half words changed
            if len(worig) / max(1,len(sorig.split())) > 0.5:
                continue
            if not (has_valid_words(worig) or\
                    has_valid_words(wmod)):
                continue
            if num_proper_nouns(sorig) > len(sorig.split()) / 2:
                continue
            #if len(deleted_new) >= 1 and len(deleted_new) < 4 and len(inserted_new) < 4:
            #data = data_old
            if len(worig) > 3 or len(wmod) > 3:
                continue
            #old_sentence = get_containing_sentence(paragraph, opr, data, original
            #        = True)
            old_sentence = sorig
            #if old_sentence[:SPAN] in old_statements_added:
            if old_sentence[:SPAN] in to_add_old:
                continue
            #new_sentence = get_containing_sentence(paragraph, opr, data, original
            #        = False)
            new_sentence = smod
            #sentences.append((old_sentence, new_sentence))
            #words_removed, words_added = get_changes_for_sentence(old_sentence, new_sentence)
            to_add.append((old_sentence, new_sentence))
            #old_statements_added.append(old_sentence[:SPAN])
            to_add_old.append(old_sentence[:SPAN])
        sentences.extend(to_add)
    return sentences


def pov_util(paragraphs, deb = False):
    label = 0
    # labeling code goes here
    
    # pov edits typically focus on one or two paragraphs
    if len(paragraphs) > 7:
        return label
    
    for paragraph in paragraphs:
        for opr, data in paragraph:
            # skip equal and insert operations
            context_before = data[0][1]
            context_after = data[-1][1]
            deleted, inserted = '', ''
            if opr == "insert":
                continue
            elif opr == "replace":
                # code for replace segments
                deleted = data[1][1]
                inserted = data[2][1]
            elif opr == "delete":
                # code for deletion segments
                deleted = data[1][1]
            elif opr == 'equal':
                continue

            # exclude reference tags
            if re.search(r'ref', inserted) or re.search(r'ref', deleted):
                continue
            # exclude category
            if 'category:' in context_before.lower() or 'category:' in inserted:
                continue
            # exclude infoboxes, lists
            # todo: whitespace, strip
            context_strip = context_before.strip()
            if context_strip.startswith('|') or context_strip.startswith('*') \
                or context_strip.startswith('#') or context_strip.startswith('='):
                if len(context_strip.split()) + len(context_after.split()) + len(inserted.split()) < 70:
                    continue
            # exclude templates
            if '{{' in context_before and not re.search(r'\{\{.+\}\}', context_before):
                continue
            # exclude full paragraphs
            if detect_newlines_util(deleted, inserted):
                continue
            # exclude wikilink modifications
            if '[[' in context_before and not re.search(r'\[\[.+\]\]', context_before):
            #if '[[' in context_before and ']]' not in context_before:
                continue
            # exclude insertions like - "|Meyrick]]", still need to handle trailing spaces
            if inserted.endswith(']]') and '[[' not in inserted:
                continue
            # exclude insertions like - "[[Edward", still need to handle leading spaces
            if '[[' in inserted or ']]' in inserted:
                continue
                # TODO: unbalanaaced in delted
            # exlude '''markup'''
            if "'''" in inserted or "'''" in deleted:
                continue
            # TODO: skip editing special characters, html tags, etc. in deleted/inserted

            # Get rid of wiki-markup
            forbidden_link_prefixes = ['category', 'image', 'file']
            wtpp = Wikitext2Words(forbidden_link_prefixes)
            inserted_stripped = wtpp.transform(inserted)    

            # skip deletions with less than a word or where the word is too short
            if len(deleted.split()) < 1 or len(deleted) < 5:
                continue
            # skip inserting numbers
            if re.search(r'\d', inserted):
                continue
            # skip deleting notes that end with }}
            if deleted.endswith('}}'):
                continue
            # skip long insertions
            if len(inserted_stripped) > 5:
                continue

            # if difference in words between deleted and inserted is less than 6, could be POV edit
            if abs(len(deleted.split()) - len(inserted_stripped)) < 6:
                # if only one word got deleted and nothing inserted and the word is not significant, skip
                if len(deleted.split()) == 1 and inserted == '':
                    sentence = get_segment_sentence(deleted, context_before, context_after)
                    if not is_significant(deleted.split(), sentence):
                        continue
            if len(deleted.split()) - len(inserted_stripped) > 5:
                label = 1
        
    return label

def is_cite(para_operations_with_context):
    paragraphs = get_observations_by_paragraph(para_operations_with_context)
    is_citation = False
    for paragraph in paragraphs:
        for opr, data in paragraph:
            data_new = None
            cb, ca = get_limited_context(data)
            _, deleted, inserted, _ = get_data_from_operations(opr, data)
            if opr == 'replace':
                data_new = (('context_before', cb), data[1], data[2], ('context_after', ca))
            else:
                data_new = (('context_before', cb), data[1], ('context_after', ca))
            data_old = data
            data = data_new
            
            # use is_only_verification to exclude citations
            for tag in cite_tags:
                if tag in inserted.lower():
                    is_citation = True
                    break
        if is_citation:
            break
    return is_citation


def small_edits_util(para_operations_with_context, comment):
    sentences = []
    old_statements_added = []
    SPAN = 15
    paragraphs = get_observations_by_paragraph(para_operations_with_context)
    if len(paragraphs) > 1 or len(paragraphs) == 0:
        return 0
    paragraph = paragraphs[0]
    valid_segs = []
    for opr, data in paragraph:
        valid = 1
        context_before, deleted, inserted, context_after = get_data_from_operations(opr, data)
        # excluide inside link modifications    
        if has_unclosed_brace(opr, data, ('[', ']')):
            valid = 0
        if has_unclosed_brace(opr, data, ('{{', '}}')):
            valid = 0
        if 'category:' in context_before.lower() or 'category:' in\
        inserted.lower():
            valid = 0
        context_strip = context_before.strip()
        # exclude infobox, table, list changes
        if context_strip.startswith('|') or context_strip.startswith('*') \
            or context_strip.startswith('#') or context_strip.startswith('=') or\
            context_before.startswith('!'):
                valid = 0
        if detect_newlines_util(deleted, inserted):
            valid = 0
        # exclude spelling fix
        # only full template, links inside
        inserted_stripped = re.sub(strip_regex, "", inserted.lower())
        deleted_stripped = re.sub(strip_regex, "", inserted.lower())
        before_words = deleted_stripped.split(' ')
        after_words = inserted_stripped.split(' ')
        if len(before_words) == 1 and len(after_words) == 1 and\
        edit_distance(deleted, inserted) <= 3: 
            valid = 0
        # exclude only symbols
        if not re.search('[a-zA-Z0-9]', inserted_stripped) and not\
        re.search('[a-zA-Z0-9]', deleted_stripped):
            valid = 0
        if 'awb' in comment.lower() or 'mos' in comment.lower():
            valid = 0
        # heading
        if '==' in context_before or '==' in inserted_stripped or '==' in context_after:
            valid = 0
        valid_segs.append(valid)
    if 1 in valid_segs:
        return 1
    else:
        return 0

