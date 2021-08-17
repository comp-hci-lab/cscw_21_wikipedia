from revscoring import Feature
from revscoring.features import wikitext
from revscoring.datasources import revision_oriented as ro
from revscoring.datasources.meta import vectorizers
from revscoring.features.meta import aggregators
from revscoring.languages import english
from revscoring import Datasource

from .datasources.diff import (relocation_segments_context,
        added_segments_context, removed_segments_context,
        operations_with_context, para_operations_with_context)
from .wiki_edit_util import (spell_error, stem_overlap, simi_overlap, user_history,
                            simi_overlap_max, simi_overlap_min, simi_overlap_avg,
                            comment_revert, comment_typo, comment_pov, is_registered,
                            gender_type, segment_length, segment_search_external,
                            segment_search_file, segment_template,
                            segment_reference, segment_internal,
                            segment_external, segment_file, segment_markup,
                            operation_in_template, operation_in_reference,
                            operation_in_internal, operation_in_external,
                            operation_in_file, is_template, is_reference,
                            is_internal, is_external, is_markup, is_file
                            )
from .edit_utils import (fact_update, seg_avg_len, seg_min_len, seg_max_len,
                        markup_chars_ratio_avg, markup_chars_ratio_min,
                        markup_chars_ratio_max, wikif, numbers_present,
                        find_elab, process, disamb, refact, clarif, simplif,
                        verif, get_segment_cats, copyedit, grammar_cats,
                        only_infobox, elab_ratio, citation_util,
                        citation_util_with_sections, clarification_util_para,
                        clarification_util_statements, pov_deletions,
                        small_edits_util, is_cite
                        )

revision_features = [	
    ro.revision.page.namespace.id,	
    ro.revision.minor,
    ro.revision.byte_len, 

    ## char features
    wikitext.revision.diff.uppercase_words_added,
    wikitext.revision.diff.chars_added,
    wikitext.revision.diff.chars_removed,
    wikitext.revision.diff.numeric_chars_added,
    wikitext.revision.diff.numeric_chars_removed,
    wikitext.revision.diff.whitespace_chars_added,
    wikitext.revision.diff.whitespace_chars_removed,
    wikitext.revision.diff.markup_chars_added,
    wikitext.revision.diff.markup_chars_removed,
    wikitext.revision.diff.cjk_chars_added,
    wikitext.revision.diff.cjk_chars_removed,
    wikitext.revision.diff.entity_chars_added,
    wikitext.revision.diff.entity_chars_removed,
    wikitext.revision.diff.url_chars_added,
    wikitext.revision.diff.url_chars_removed,
    wikitext.revision.diff.word_chars_added,
    wikitext.revision.diff.word_chars_removed,
    wikitext.revision.diff.uppercase_words_added,
    wikitext.revision.diff.uppercase_words_removed,
    wikitext.revision.diff.punctuation_chars_added,
    wikitext.revision.diff.punctuation_chars_removed,
    ## token features
    wikitext.revision.diff.token_delta_sum,
    wikitext.revision.diff.token_delta_increase,
    wikitext.revision.diff.token_delta_decrease,
    wikitext.revision.diff.token_prop_delta_sum,
    wikitext.revision.diff.token_prop_delta_increase,
    wikitext.revision.diff.token_prop_delta_decrease,
    wikitext.revision.diff.number_delta_sum,
    wikitext.revision.diff.number_delta_increase,
    wikitext.revision.diff.number_delta_decrease,
    wikitext.revision.diff.number_prop_delta_sum,
    wikitext.revision.diff.number_prop_delta_increase,
    wikitext.revision.diff.number_prop_delta_decrease,
    wikitext.revision.diff.whitespace_delta_sum,
    wikitext.revision.diff.whitespace_delta_increase,
    wikitext.revision.diff.whitespace_delta_decrease,
    wikitext.revision.diff.whitespace_prop_delta_sum,
    wikitext.revision.diff.whitespace_prop_delta_increase,
    wikitext.revision.diff.whitespace_prop_delta_decrease,
    wikitext.revision.diff.markup_delta_sum,
    wikitext.revision.diff.markup_delta_increase,
    wikitext.revision.diff.markup_delta_decrease,
    wikitext.revision.diff.markup_prop_delta_sum,
    wikitext.revision.diff.markup_prop_delta_increase,
    wikitext.revision.diff.markup_prop_delta_decrease,
    wikitext.revision.diff.cjk_delta_sum,
    wikitext.revision.diff.cjk_delta_increase,
    wikitext.revision.diff.cjk_delta_decrease,
    wikitext.revision.diff.cjk_prop_delta_sum,
    wikitext.revision.diff.cjk_prop_delta_increase,
    wikitext.revision.diff.cjk_prop_delta_decrease,
    wikitext.revision.diff.entity_delta_sum,
    wikitext.revision.diff.entity_delta_increase,
    wikitext.revision.diff.entity_delta_decrease,
    wikitext.revision.diff.entity_prop_delta_sum,
    wikitext.revision.diff.entity_prop_delta_increase,
    wikitext.revision.diff.entity_prop_delta_decrease,
    wikitext.revision.diff.url_delta_sum,
    wikitext.revision.diff.url_delta_increase,
    wikitext.revision.diff.url_delta_decrease,
    wikitext.revision.diff.url_prop_delta_sum,
    wikitext.revision.diff.url_prop_delta_increase,
    wikitext.revision.diff.url_prop_delta_decrease,
    wikitext.revision.diff.word_delta_sum,
    wikitext.revision.diff.word_delta_increase,
    wikitext.revision.diff.word_delta_decrease,
    wikitext.revision.diff.word_prop_delta_sum,
    wikitext.revision.diff.word_prop_delta_increase,
    wikitext.revision.diff.word_prop_delta_decrease,
    wikitext.revision.diff.uppercase_word_delta_sum,
    wikitext.revision.diff.uppercase_word_delta_increase,
    wikitext.revision.diff.uppercase_word_delta_decrease,
    wikitext.revision.diff.uppercase_word_prop_delta_sum,
    wikitext.revision.diff.uppercase_word_prop_delta_increase,
    wikitext.revision.diff.uppercase_word_prop_delta_decrease,
    wikitext.revision.diff.punctuation_delta_sum,
    wikitext.revision.diff.punctuation_delta_increase,
    wikitext.revision.diff.punctuation_delta_decrease,
    wikitext.revision.diff.punctuation_prop_delta_sum,
    wikitext.revision.diff.punctuation_prop_delta_increase,
    wikitext.revision.diff.punctuation_prop_delta_decrease,
    wikitext.revision.diff.break_delta_sum,
    wikitext.revision.diff.break_delta_increase,
    wikitext.revision.diff.break_delta_decrease,
    wikitext.revision.diff.break_prop_delta_sum,
    wikitext.revision.diff.break_prop_delta_increase,
    wikitext.revision.diff.break_prop_delta_decrease,
    ## token edit features
    wikitext.revision.diff.segments_added,
    wikitext.revision.diff.segments_removed,
    wikitext.revision.diff.tokens_added,
    wikitext.revision.diff.tokens_removed,
    wikitext.revision.diff.numbers_added,
    wikitext.revision.diff.numbers_removed,
    wikitext.revision.diff.markups_added,
    wikitext.revision.diff.markups_removed,
    wikitext.revision.diff.whitespaces_added,
    wikitext.revision.diff.whitespaces_removed,
    wikitext.revision.diff.cjks_added,
    wikitext.revision.diff.cjks_removed,
    wikitext.revision.diff.entities_added,
    wikitext.revision.diff.entities_removed,
    wikitext.revision.diff.urls_added,
    wikitext.revision.diff.urls_removed,
    wikitext.revision.diff.words_added,
    wikitext.revision.diff.words_removed,
    wikitext.revision.diff.uppercase_words_added,
    wikitext.revision.diff.uppercase_words_removed,
    wikitext.revision.diff.punctuations_added,
    wikitext.revision.diff.punctuations_removed,
    wikitext.revision.diff.breaks_added,
    wikitext.revision.diff.breaks_removed,
    wikitext.revision.diff.longest_token_added,
    wikitext.revision.diff.longest_uppercase_word_added,

    # *** language features 
    # *** stop word features 
    english.stopwords.revision.diff.stopwords_added,
    english.stopwords.revision.diff.stopwords_removed,
    english.stopwords.revision.diff.non_stopwords_added,
    english.stopwords.revision.diff.non_stopwords_removed,
    english.stopwords.revision.diff.stopword_delta_sum,
    english.stopwords.revision.diff.stopword_delta_increase,
    english.stopwords.revision.diff.stopword_delta_decrease,
    english.stopwords.revision.diff.non_stopword_delta_sum,
    english.stopwords.revision.diff.non_stopword_delta_increase,
    english.stopwords.revision.diff.non_stopword_delta_decrease,
    english.stopwords.revision.diff.stopword_prop_delta_sum,
    english.stopwords.revision.diff.stopword_prop_delta_increase,
    english.stopwords.revision.diff.stopword_prop_delta_decrease,
    english.stopwords.revision.diff.non_stopword_prop_delta_sum,
    english.stopwords.revision.diff.non_stopword_prop_delta_increase,
    english.stopwords.revision.diff.non_stopword_prop_delta_decrease,

    # *** stemmed features 
    english.stemmed.revision.diff.stem_delta_sum,
    english.stemmed.revision.diff.stem_delta_increase,
    english.stemmed.revision.diff.stem_delta_decrease,
    english.stemmed.revision.diff.stem_prop_delta_sum,
    english.stemmed.revision.diff.stem_prop_delta_increase,
    english.stemmed.revision.diff.stem_prop_delta_decrease,

    # *** badwords 
    english.badwords.revision.diff.matches_added,
    english.badwords.revision.diff.matches_removed,
    english.badwords.revision.diff.match_delta_sum,
    english.badwords.revision.diff.match_delta_increase,
    english.badwords.revision.diff.match_delta_decrease,
    english.badwords.revision.diff.match_prop_delta_sum,
    english.badwords.revision.diff.match_prop_delta_increase,
    english.badwords.revision.diff.match_prop_delta_decrease,

    # *** informals
    english.informals.revision.diff.matches_added,
    english.informals.revision.diff.matches_removed,
    english.informals.revision.diff.match_delta_sum,
    english.informals.revision.diff.match_delta_increase,
    english.informals.revision.diff.match_delta_decrease,
    english.informals.revision.diff.match_prop_delta_sum,
    english.informals.revision.diff.match_prop_delta_increase,
    english.informals.revision.diff.match_prop_delta_decrease
]

feature_spell_error_segment_added = Feature(
    "spell_error_segment_added",
    spell_error,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=float
)

feature_spell_error_segment_removed = Feature(
    "spell_error_segment_removed",
    spell_error,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=float
)

feature_stem_overlap_segments = Feature(
    "stem_overlap_segments",
    stem_overlap,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                wikitext.revision.diff.datasources.segments_removed],
    returns=int
)

feature_simi_overlap_segments = Datasource(
    "simi_overlap_segments",
    simi_overlap,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                wikitext.revision.diff.datasources.segments_removed]
)

feature_simi_overlap_max = Feature(
    "simi_overlap_max",
    simi_overlap_max,
    depends_on=[feature_simi_overlap_segments],
    returns=float
)

feature_simi_overlap_min = Feature(
    "simi_overlap_min",
    simi_overlap_min,
    depends_on=[feature_simi_overlap_segments],
    returns=float
)

feature_simi_overlap_avg = Feature(
    "simi_overlap_avg",
    simi_overlap_avg,
    depends_on=[feature_simi_overlap_segments],
    returns=float
)

feature_user_history = Feature(
    "user_history_registration",
    user_history,
    depends_on=[ro.revision.timestamp,
                ro.revision.user.info.registration],
    returns=int
)

feature_comment_revert = Feature(
    "comment_revert",
    comment_revert,
    depends_on=[ro.revision.comment],
    returns=int
)

feature_comment_length = aggregators.len(
    ro.revision.comment,
    name="comment_length",
    returns=int
)

feature_comment_typo = Feature(
    "comment_typo",
    comment_typo,
    depends_on=[ro.revision.comment],
    returns=int
)

feature_comment_pov = Feature(
    "comment_pov",
    comment_pov,
    depends_on=[ro.revision.comment],
    returns=int
)

feature_reloc_len = aggregators.len(
    relocation_segments_context,
    name="relocation_length",
    returns=int
)

feature_is_registered = Feature(
    "is_registered",
    is_registered,
    depends_on=[ro.revision.user.info.registration],
    returns=int
)

feature_gender_type = Feature(
    "gender_type",
    gender_type,
    depends_on=[ro.revision.user.info.gender],
    returns=int
)

feature_segment_length_added = Feature(
    "segment_length_added",
    segment_length,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)
feature_segment_length_removed = Feature(
    "segment_length_removed",
    segment_length,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int
)

feature_segment_search_external_added = Feature(
    "segment_search_external_added",
    segment_search_external,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)

feature_segment_search_external_removed = Feature(
    "segment_search_external_removed",
    segment_search_external,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int
)

feature_segment_search_file_added = Feature(
    "segment_search_file_added",
    segment_search_file,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)
feature_segment_search_file_removed = Feature(
    "segment_search_file_removed",
    segment_search_file,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int
)

feature_segment_template_added = Feature(
    "segment_template_added",
    segment_template,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)
feature_segment_template_removed = Feature(
    "segment_template_removed",
    segment_template,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int
)

feature_segment_reference_added = Feature(
    "segment_reference_added",
    segment_reference,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)
feature_segment_reference_removed = Feature(
    "segment_reference_removed",
    segment_reference,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int
)

feature_segment_internal_added = Feature(
    "segment_internal_added",
    segment_internal,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)
feature_segment_internal_removed = Feature(
    "segment_internal_removed",
    segment_internal,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int
)

feature_segment_external_added = Feature(
    "segment_external_added",
    segment_external,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)
feature_segment_external_removed = Feature(
    "segment_external_removed",
    segment_external,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int
)

feature_segment_file_added = Feature(
    "segment_file_added",
    segment_file,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)
feature_segment_file_removed = Feature(
    "segment_file_removed",
    segment_file,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int)

feature_segment_markup_added = Feature(
    "segment_markup_added",
    segment_markup,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=int
)
feature_segment_markup_removed = Feature(
    "segment_markup_removed",
    segment_markup,
    depends_on=[wikitext.revision.diff.datasources.segments_removed],
    returns=int)

feature_operation_in_template_added = Feature(
    "operation_in_template_added",
    operation_in_template,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_operation_in_template_removed = Feature(
    "operation_in_template_removed",
    operation_in_template,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_operation_in_reference_added = Feature(
    "operation_in_reference_added",
    operation_in_reference,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_operation_in_reference_removed = Feature(
    "operation_in_reference_removed",
    operation_in_reference,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_operation_in_internal_added = Feature(
    "operation_in_internal_added",
    operation_in_internal,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_operation_in_internal_removed = Feature(
    "operation_in_internal_removed",
    operation_in_internal,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_operation_in_external_added = Feature(
    "operation_in_external_added",
    operation_in_external,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_operation_in_external_removed = Feature(
    "operation_in_external_removed",
    operation_in_external,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_operation_in_file_added = Feature(
    "operation_in_file_added",
    operation_in_file,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_operation_in_file_removed = Feature(
    "operation_in_file_removed",
    operation_in_file,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_is_template_added = Feature(
    "is_template_added",
    is_template,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_is_template_removed = Feature(
    "is_template_removed",
    is_template,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_is_reference_added = Feature(
    "is_reference_added",
    is_reference,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_is_reference_removed = Feature(
    "is_reference_removed",
    is_reference,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_is_internal_added = Feature(
    "is_internal_added",
    is_internal,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_is_internal_removed = Feature(
    "is_internal_removed",
    is_internal,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_is_external_added = Feature(
    "is_external_added",
    is_external,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_is_external_removed = Feature(
    "is_external_removed",
    is_external,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_is_markup_added = Feature(
    "is_markup_added",
    is_markup,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_is_markup_removed = Feature(
    "is_markup_removed",
    is_markup,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_is_file_added = Feature(
    "is_file_added",
    is_file,
    depends_on=[wikitext.revision.diff.datasources.segments_added,
                added_segments_context],
    returns=int
)
feature_is_file_removed = Feature(
    "is_file_removed",
    is_file,
    depends_on=[wikitext.revision.diff.datasources.segments_removed,
                removed_segments_context],
    returns=int
)

feature_seg_avglen = Feature(
    "segment_avg_len",
    seg_avg_len,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=float
)

feature_seg_minlen = Feature(
    "segment_avg_len",
    seg_min_len,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=float
)

feature_seg_maxlen = Feature(
    "segment_avg_len",
    seg_max_len,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=float
)

feature_seg_markup_ratio_avg = Feature(
    "segment_markup_ratio_avg",
    markup_chars_ratio_avg,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=float
)

feature_seg_markup_ratio_min = Feature(
    "segment_markup_ratio_min",
    markup_chars_ratio_min,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=float
)

feature_seg_markup_ratio_max = Feature(
    "segment_markup_ratio_max",
    markup_chars_ratio_max,
    depends_on=[wikitext.revision.diff.datasources.segments_added],
    returns=float
)


feature_wikif = Feature(
    "wikif",
    wikif,
    depends_on=[operations_with_context],
    returns=int
)

feature_max_num_ratio = Feature(
        "numbers_ratio",
        numbers_present,
        depends_on=[operations_with_context],
        returns=float
        )

feature_fact_update = Feature(
    "fact_update",
    fact_update,
    depends_on=[operations_with_context],
    returns=int
)


feature_elab = Feature(
        "elab",
        find_elab,
        depends_on=[operations_with_context],
        returns=int
        )

feature_process = Feature(
        "process",
        process,
        depends_on=[operations_with_context],
        returns=int
        )

feature_disamb = Feature(
        "disamb",
        disamb,
        depends_on=[operations_with_context],
        returns=int
        )

feature_refact = Feature(
        "refact",
        refact,
        depends_on=[operations_with_context],
        returns=int
        )

feature_clarif = Feature(
        "clarif",
        clarification_util_para,
        depends_on=[para_operations_with_context],
        returns=int
        )

feature_small_edits = Feature(
        "small_edits",
        small_edits_util,
        depends_on=[para_operations_with_context, ro.revision.comment],
        returns=int
        )

feature_simplif = Feature(
        "simplif",
        simplif,
        depends_on=[operations_with_context],
        returns=int
        )

feature_verif = Feature(
        "verif",
        verif,
        depends_on=[operations_with_context],
        returns=int
        )

feature_copyed = Feature(
        "copyedit",
        copyedit,
        depends_on=[operations_with_context],
        returns=int
        )

feature_only_infobox = Feature(
        "only_infobox",
        only_infobox,
        depends_on=[operations_with_context],
        returns=int
        )

segment_categories = Datasource(
        "segment_categories", get_segment_cats,
        depends_on=[operations_with_context]
        )

grammar_cats = Datasource(
        "grammar_cats", grammar_cats,
        depends_on = [operations_with_context]
        )

feature_elab_ratio = Feature(
        "elab_ratio",
        elab_ratio,
        depends_on = [operations_with_context],
        returns = float
        )

citation_ds = Datasource(
        "citation",
        citation_util,
        depends_on = [operations_with_context],
        )

citation_ds_sections = Datasource(
        "citation_sections",
        citation_util_with_sections,
        depends_on = [wikitext.revision.datasources.diff.operations,]
        )

feat_citation = Feature(
        "feature_citation",
        is_cite,
        depends_on = [para_operations_with_context],
        returns=int
        )

clarification_ds = Datasource(
        "clarification",
        clarification_util_statements,
        depends_on = [para_operations_with_context],
        )

pov_deletions_ds = Datasource(
        "pov_deletions",
        pov_deletions,
        depends_on = [para_operations_with_context],
        )

text_features = [
    feature_spell_error_segment_added,
    feature_spell_error_segment_removed,
    feature_stem_overlap_segments,
    #feature_simi_overlap_min,
    #feature_simi_overlap_max,
    #feature_simi_overlap_avg,
    feature_user_history,
    feature_comment_revert,
    feature_comment_length,
    feature_comment_typo,
    feature_comment_pov,
    feature_reloc_len,
    feature_is_registered,
    feature_gender_type,
    feature_segment_length_added,
    feature_segment_length_removed,
    feature_segment_search_external_added,
    feature_segment_search_external_removed,
    feature_segment_search_file_added,
    feature_segment_search_file_removed,
    feature_segment_template_added,
    feature_segment_template_removed,
    feature_segment_reference_added,
    feature_segment_reference_removed,
    feature_segment_internal_added,
    feature_segment_internal_removed,
    feature_segment_external_added,
    feature_segment_external_removed,
    feature_segment_file_added,
    feature_segment_file_removed,
    feature_segment_markup_added,
    feature_segment_markup_removed,
    feature_operation_in_template_added,
    feature_operation_in_template_removed,
    feature_operation_in_reference_added,
    feature_operation_in_reference_removed,
    feature_operation_in_internal_added,
    feature_operation_in_internal_removed,
    feature_operation_in_external_added,
    feature_operation_in_external_removed,
    feature_operation_in_file_added,
    feature_operation_in_file_removed,
    feature_is_template_added,
    feature_is_template_removed,
    feature_is_reference_added,
    feature_is_reference_removed,
    feature_is_internal_added,
    feature_is_internal_removed,
    feature_is_external_added,
    feature_is_external_removed,
    feature_is_markup_added,
    feature_is_markup_removed,
    feature_is_file_added,
    feature_is_file_removed,
##########
    #feature_seg_avglen,
    #feature_seg_maxlen,
    #feature_seg_minlen,
    #feature_fact_update,
    #feature_seg_markup_ratio_avg,
    #feature_seg_markup_ratio_min,
    #feature_seg_markup_ratio_max,
    #feature_max_num_ratio,
]

rule_features = [    
    feature_wikif,
    feature_elab,
    feature_process,
    feature_disamb,
    feature_refact,
    feature_clarif,
    feature_simplif,
    feature_verif,
    feature_copyed,
    feature_only_infobox
]

pov_deletions_ds = [pov_deletions_ds]
clarification_ds = [clarification_ds]
feature_citation = [feat_citation]

wikif = [feature_wikif]

fup = [feature_fact_update]

elab = [feature_elab]

process = [feature_process]

disamb = [feature_disamb]

refact = [feature_refact]

clarif = [feature_clarif]

simplif = [feature_simplif]

verif = [feature_verif]

copyed = [feature_copyed]

small_edits = [feature_small_edits]

edittypes = revision_features + text_features

edittypes = revision_features + text_features
#edittypes += rule_features
