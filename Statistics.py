from manim import *
from Constants_helper import *
import numpy as np
import scipy.stats as sp
from scipy.special import gamma

#python -m manim -p -ql Statistics.py Hypothesis


class Density:

    def pdf(self, x):
        return self.density.pdf(x, loc=self.loc, scale=self.scale)

    def cdf(self, x):
        return self.density.cdf(x, loc=self.loc, scale=self.scale)

    def qdf(self, x):
        # inverse cdf
        return self.density.ppf(x, loc=self.loc, scale=self.scale)

    def RV(self, n):
        # returns sample of n iid random variables
        return self.density.rvs(loc=self.loc, scale=self.scale, size=n)

    def CI(self, alpha=0.05, side=2, standardized=True):
        '''
        :param alpha: significant level
        :param standardized: boolean if CI is standard normal or not
        :param side: "left", "right", or 2 sided test
        :return: Confidence interval
        '''
        assert type(standardized) == bool, "standardized boolean is not boolean"
        assert side in [2, "left", "right"], "Incorrect input for side"
        # 2 side need tails to sum to alpha, 1 side doesnt
        if side == 2:
            # half area on each side
            alpha = alpha / 2

        if standardized:
            left = self.density.ppf(alpha)
            right = self.density.ppf(1 - alpha)
        else:  # not standardized
            left = self.density.ppf(alpha, loc=self.loc, scale=self.scale)
            right = self.density.ppf(1 - alpha, loc=self.loc, scale=self.scale)

        if side == "left":
            return np.array([left, np.inf])
        elif side == "right":
            return np.array([np.inf, right])
        else:
            return np.array([left, right])

    def p_val(self, TS, side, standardized=True):
        '''
        Probability of observing the Test Statistic (or greater/less than) given the null is true
        if p_val < alpha then reject
        else accept
        '''
        assert side in [2, "left", "right"], "Incorrect input for side"

        if standardized:
            p = self.density.cdf(TS)
        else:
            p = self.cdf(TS)
        if side == "left":
            return p
        if side == "right":
            return 1 - p
        else:
            if TS < 0:
                return 2 * p
            else:
                return 2 * (1 - p)


class Uniform(Density):
    # Continuous Uniform

    def __init__(self, a, b):
        self.density = sp.uniform
        # Distribution parameters
        self.a = a
        self.b = b

        # distribution statistics
        self.mean = (a + b) / 2
        self.var = (b - a) ** 2 / 12
        self.sd = self.var ** 0.5

        # scipy parameters
        self.loc = a
        self.scale = b - a


class T(Density):
    # unshifted T distribution

    def __init__(self, df):
        self.density = sp.t
        # Distribution parameters
        self.df = df  # any real number

        # distribution statistics
        if df > 1:
            self.mean = 0
        else:
            print("Warning: Given df provides an undefined mean")
            self.mean = None
        if df > 2:
            self.var = df / (df - 2)
        elif 1 < df and df <= 2:
            self.var = np.inf
        else:
            print("Warning: Given df provides an undefined variance")
        if df > 1:
            self.sd = self.var ** 0.5

        # scipy parameters
        self.loc = 0
        self.scale = 1


class Normal(Density):
    def __init__(self, mu, sigma2):
        self.density = sp.norm

        # Distribution parameters
        self.mu = mu
        self.sigma2 = sigma2

        # distribution statistics
        self.mean = mu
        self.var = sigma2
        self.sd = sigma2 ** 0.5

        # scipy parameters
        self.loc = self.mean
        self.scale = self.sd


class Chi2(Density):
    def __init__(self, df):
        self.density = sp.chi2

        # Distribution parameters
        self.df = df

        # distribution statistics
        self.mean = df
        self.var = 2 * df
        self.sd = self.var ** 0.5

        # scipy parameters
        self.loc = 0
        self.scale = 1


class Z_test_1_samp(Scene):
    # python -m manim -p -ql Statistics.py Z_test_1_samp
    def construct(self):
        intro = MathTex(r"\mu = \text{Population mean}")

        self.add(intro)



class Hypothesis(Scene):
    #python -m manim -p -ql Statistics.py Hypothesis
    def construct(self):

        ax = Axes(
            x_range=[-3,3, 0.5], y_range=[0,0.5, 0.1],
            x_length=10,         y_length=6,
            axis_config={"include_tip": True},
        ).add_coordinates()
        labels = ax.get_axis_labels()

        mu = 170
        sigma = 5
        n = 10000
        alpha = 0.05
        gen_data = sp.norm.rvs(loc=mu, scale=sigma, size=n)
        sample_mean = np.mean(gen_data)
        variable = Normal(0, 1)
        bounds = variable.CI(n, alpha, True, 2)

        density = ax.get_graph(lambda x: variable.pdf(x), x_range=[-3, 3], color=RED)

        mean = ax.get_vertical_line(ax.input_to_graph_point((sample_mean-mu)/sigma, density), color=YELLOW)


        #Confidence Interval bounds
        bound0 = ax.get_vertical_line(ax.input_to_graph_point(bounds[0], density), color=YELLOW)
        bound1 = ax.get_vertical_line(ax.input_to_graph_point(bounds[1], density), color=YELLOW)

        #Area under the curve
        area1 = ax.get_area(density, x_range=[-3,bounds[0]], color=GREY, opacity=0.2)
        area2 = ax.get_area(density, x_range=[bounds[1],3], color=GREY, opacity=0.2)

        self.play(
            DrawBorderThenFill(ax),
            DrawBorderThenFill(labels),
            DrawBorderThenFill(density),
            Write(bound0),
            Write(bound1),
        )
        self.wait()

        self.play(
            Write(area1, run_time=0.5),
            Write(area2, run_time=0.5),
            Write(mean)
        )
        self.wait()
