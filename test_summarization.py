import unittest
from summarization import Summarization
from utility import Utility

class TestSummarization(unittest.TestCase):

    def setUp(self):
        self.short_text = "Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal."
        self.long_text = "The world's oldest football clubs were founded in England from 1789 and, in the 1871–72 season, the FA "\
                         "Cup was founded as the world's biggest first organised competition. The first international match took "\
                         "place in November 1872 when England travelled to Glasgow to play Scotland. The quality of Scottish players "\
                         "was such that northern English clubs began offering them professional terms to move south. At first, the FA "\
                         "was strongly opposed to professional and that gave rise to a bitter dispute from 1880 until the FA relented "\
                         "and formally legitimised professionalism in 1885. A shortage of competitive matches led to the formation of "\
                         "the Football League by twelve professional clubs in 1888 and the domestic game has ever since then been based "\
                         "on the foundation of league and cup football. The competitiveness of matches involving professional teams "\
                         "generated widespread interest, especially amongst the working class. Attendances increased significantly through "\
                         "the 1890s and the clubs had to build larger grounds to accommodate them. Typical ground construction was mostly "\
                         "terracing for standing spectators with limited seating provided in a grandstand built centrally alongside one of "\
                         "the pitch touchlines. Through media coverage, football became a main talking point among the population and had "\
                         "overtaken cricket as England's national sport by the early 20th century. The size of the Football League increased "\
                         "from the original twelve clubs in 1888 to 92 in 1950. The clubs were organised by team merit in four divisions with "\
                         "promotion and relegation atthe end of each season. Internationally, England hosted and won the 1966 FIFA World Cup "\
                         "but has otherwise been among the also-rans in global terms. English clubs have been a strong presence in European "\
                         "competition with several teams, especially Liverpool and Manchester United, winning the major continental trophies."


    def tearDown(self):
        pass

    def test_lsa_method_on_short_text(self):
        s = Summarization(lang_code="en", method="LSA")
        pred = s.summarize(self.short_text, n_sents=1)
        self.assertTrue(isinstance(pred, str))
        self.assertEqual(pred, "required at least 200 words")

    def test_lsa_method_on_long_text(self):
        n_words = Utility.get_doc_length(self.long_text)
        self.assertTrue(n_words > 200)
        s = Summarization(lang_code="en", method="LSA")
        pred = s.summarize(self.long_text, n_sents=5)
        self.assertTrue(isinstance(pred, dict))
        self.assertEqual(pred["message"], "successful")
    
    def test_text_rank_method_on_short_text(self):
        n_words = Utility.get_doc_length(self.short_text)
        self.assertTrue(n_words < 200)
        s = Summarization(lang_code="en", method="TEXT_RANK")
        pred = s.summarize(self.short_text, n_sents=1)
        self.assertTrue(isinstance(pred, str))
        self.assertEqual(pred, "required at least 200 words")
    
    def test_text_rank_method_on_long_text(self):
        n_words = Utility.get_doc_length(self.long_text)
        self.assertTrue(n_words > 200)
        s = Summarization(lang_code="en", method="TEXT_RANK")
        pred = s.summarize(self.long_text, n_sents=5)
        self.assertTrue(isinstance(pred, dict))
        self.assertEqual(pred["message"], "successful")

    def test_invalid_language(self):
        s = Summarization(lang_code="fr", method="TEXT_RANK")
        pred = s.summarize(self.long_text, n_sents=5)
        self.assertTrue(isinstance(pred, str))
        self.assertEqual(pred, "language not supported")