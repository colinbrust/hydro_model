from scripts import modelGen
import nose


class TestGenModelParams(object):
    @classmethod
    def setup_class(cls):
        """This method is run once for each class before any tests are run"""
        f = "../params/param_files_test.json"
        cls.size = 1000
        cls.param_gen_obj = modelGen.GenModelParams(f, cls.size)

    @classmethod
    def teardown_class(cls):
        """This method is run once for each class _after_ all tests are run"""

    def setUp(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    def test__get_raster_values(self):

        df = TestGenModelParams.param_gen_obj._get_raster_values()
        nose.tools.assert_equals(df.shape[0], TestGenModelParams.size)


    def test_set_rand_array(self):

        TestGenModelParams.param_gen_obj.set_rand_array(5)
        nose.tools.assert_equals(TestGenModelParams.param_gen_obj.rand_array, 5)

