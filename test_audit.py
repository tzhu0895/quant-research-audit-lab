import unittest
import numpy as np
from audit import moving_block_mean_ci,newey_west_mean_se,purged_split,rank_ic
class TestAudit(unittest.TestCase):
 def test_constant_interval(self):
  a=moving_block_mean_ci(np.ones(20),3,100);self.assertEqual((a["lower"],a["upper"]),(1,1))
 def test_deterministic(self):
  self.assertEqual(moving_block_mean_ci(np.arange(20),3,100),moving_block_mean_ci(np.arange(20),3,100))
 def test_purge(self):
  a,b=purged_split(np.arange(10),np.arange(10)+2,7);self.assertEqual(a.tolist(),[0,1,2,3,4]);self.assertEqual(b.tolist(),[7,8,9])
 def test_hac_constant(self): self.assertEqual(newey_west_mean_se(np.ones(20),3),0)
 def test_rank_ic(self): self.assertAlmostEqual(rank_ic([1,2,3],[3,2,1]),-1)
 def test_bad_input(self):
  with self.assertRaises(ValueError):moving_block_mean_ci([1,float("nan")])
if __name__=="__main__":unittest.main()
