'''
Module for performing k-means on data on a 2-torus. For full documentation 
see toroidalkmeans.py.

This file is part of toroidal k-means.

Toroidal k-means is free software: you can redistribute 
it and/or modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation, either version 3 of the License, 
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

author: Simon Wilshin
contact: swilshin@gmail.com
date: Jan 2016
'''

from toroidalkmeans import (g2T,g2,euclidDistFunc,quotient2TorusDistFunc,
  kstep,torkmeans)