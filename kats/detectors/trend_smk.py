# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import logging
from datetime import datetime
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pymannkendall as mk

    _no_mk = False
except ImportError:
    _no_mk = True
from kats.consts import TimeSeriesChangePoint, TimeSeriesData
from kats.detectors.detector import Detector
from statsmodels.tsa.api import SimpleExpSmoothing

"""Seasonal Mann-Kendall (SMK) Trend Detector Module

This module includes detectors based on the Seasonal Mann-Kendall Test, which is a
non-parametric test for monotonic trends.
"""


class SMKChangePoint(TimeSeriesChangePoint):
    """Changepoint for SMKDetector

    Attributes:

        start_time: Start time of the change.
        end_time: End time of the change.
        confidence: The confidence of the change point.
        detector_type: Detector, Type of detector changepoint is for.  Right
            now, this is always SMKDetector.
        is_multivariate: boolean, Whether this is a changepoint for a
            multivariate time series.
        trend_direction: string, Direction of trend, either 'increasing' or
            'decreasing'.
        Tau: float or Dict, Kendall's Tau value for changepoint.
            This is a float in the univariate case and a Dict in the
                multivariate case.
    """

    def __init__(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        confidence: float,
        is_multivariate: bool,
        trend_direction: str,
        Tau: Union[float, Dict],
    ):
        super().__init__(start_time, end_time, confidence)
        if _no_mk:
            raise RuntimeError("requires pymannkendall to be installed")
        self._detector_type = SMKDetector
        self._is_multivariate = is_multivariate
        self._trend_direction = trend_direction
        self._Tau = Tau

    @property
    def detector_type(self) -> Type:
        return self._detector_type

    @property
    def is_multivariate(self) -> bool:
        return self._is_multivariate

    @property
    def trend_direction(self) -> str:
        return self._trend_direction

    @property
    def Tau(self) -> Union[float, Dict]:
        return self._Tau  # Tau is a dict in multivariate case

    def __repr__(self) -> str:
        return (
            f"SMKChangePoint(start_time: {self._start_time}, end_time: "
            f"{self._end_time}, confidence: {self._confidence}, "
            f"detector_type: {self._detector_type}, is_multivariate: "
            f"{self._is_multivariate}, trend_direction: {self._trend_direction} "
            f"Tau: {self._Tau})"
        )


class SMKDetector(Detector):
    """
    SMKDetector (SMK stands for Seasonal Mann-Kendall) is a non-parametric statistical test
    used to determine whether there is a monotonic trend in a given time series.

    The basic idea is to check whether there is a monotonic trend based on a
    look back number of time steps (`window_size`).

    Parameters:

        data: `TimeSeriesData`, this is time series data at one-day granularity.
            This time series can be either univariate or multivariate.
            We require more than training_days points in each time series.
        threshold: `float`, threshold for trend intensity; higher threshold gives
            trend with high intensity (0.8 by default).  If we only want to use
            the p-value to determine changepoints, set threshold = 0.
        alpha: `float`, significance level (0.05 by default)
        multivariate: `bool`, whether the input time series is multivariate
    """

    window_size: Optional[int] = None
    training_days: Optional[int] = None
    direction: Optional[str] = None
    freq: Optional[str] = None
    ts: Optional[pd.DataFrame] = None
    SMK_statistics: Optional[pd.DataFrame] = None

    def __init__(
        self,
        data: Optional[TimeSeriesData] = None,
        threshold: float = 0.8,
        alpha: float = 0.05,
        multivariate: bool = False,
    ) -> None:
        # pyre-fixme[6]: Expected `TimeSeriesData` for 1st param but got
        #  `Optional[TimeSeriesData]`.
        super(SMKDetector, self).__init__(data=data)
        if _no_mk:
            raise RuntimeError("requires pymannkendall to be installed")

        self.threshold = threshold
        self.alpha = alpha
        self.multivariate = multivariate
        self.__subtype__ = "trend_detector"

        # Assume univariate but multivariate data is detected
        if self.data is not None:
            if not self.data.is_univariate() and not self.multivariate:
                logging.warning("Using multivariate SMK test for univariate data.")
                self.multivariate = True
            # Assume multivariate but univariate data is detected
            elif self.data.is_univariate() and self.multivariate:
                logging.warning("Using univariate SMK test on multivariate data.")

    def _remove_seasonality(
        self, ts: pd.DataFrame, freq: Optional[str] = None
    ) -> pd.DataFrame:
        """Remove seasonality in the time series using moving average."""

        if freq is None:
            return ts  # no seasonality
        else:
            map = {"weekly": 7, "monthly": 30, "yearly": 365}
            ts = ts.rolling(window=map[freq]).mean()
        return ts

    def _smoothing(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Remove noise in the time series using Holt-Winters model."""

        smoothed_ts = pd.DataFrame()
        for c in ts.columns:
            ts_c = ts[c].dropna()
            assert ts_c is not None
            with np.errstate(divide="raise"):
                try:
                    model = SimpleExpSmoothing(ts_c)
                    _fit = model.fit(smoothing_level=0.2, optimized=False)
                    smoothed_ts_tmp = _fit.predict(
                        start=ts_c.index[0],
                        end=ts_c.index[-1],
                    )
                    smoothed_ts = pd.concat(
                        [smoothed_ts, smoothed_ts_tmp.rename(c)], axis=1
                    )
                except FloatingPointError:
                    smoothed_ts = pd.concat([smoothed_ts, ts_c], axis=1)
                    logging.debug(
                        "Your data does not have noise. No need for smoothing"
                    )

        return smoothed_ts

    def _preprocessing(self, ts: pd.DataFrame) -> Tuple[np.ndarray, int]:
        """Check and convert the dataframe ts to an numpy array.

        Args:
            ts: a time series dataframe with time as index.

        Returns:
            A numpy array of length n and the # of metrics.
        """
        window_size = self.window_size
        assert window_size is not None

        # takes only window_size days
        x = np.asarray(ts[-window_size:])
        dim = x.ndim

        # checks the dimension of the data
        if dim == 2:  # dim should always be 2
            (n, c) = x.shape  # n is # of obs = window_size; c is # of metrics
            if c == 1:  # univariate case
                dim = 1  # convert x from 2-dim array (n, 1) to 1-dim array (n,)
                x = x.flatten()
        else:
            msg = f"dim = 2 is expected but your data has dim = {dim}."
            raise ValueError(msg)

        return x, c

    def _drop_missing_values(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Drop the missing values in x."""

        if x.ndim == 1:  # univariate case with 1-dim array/ shape(n,)
            x = x[~(np.isnan(x))]
        else:  # multivariate case with 2-dim arrat/ shape (n, c)
            x = x[~np.isnan(x).any(axis=1)]
        return x, len(x)

    def _apply_threshold(self, trend, Tau):
        if abs(Tau) <= self.threshold:
            return "no trend"
        return trend

    def SMKtest(self, ts: pd.DataFrame, period: int) -> Tuple[datetime, str, float, float]:
        """Performs the Seasonal Mann-Kendall (SMK) test for trend detection.

        Args:
            ts: the dataframe of input data with time as index.

        Returns:
            (tuple): tuple containing:

                anchor_date(datetime): the last time point in ts; the date for
                    which alert is triggered
                trend(str): tells the trend (decreasing, increasing, or no trend)
                p(float): p-value of the significance test
                Tau(float): Kendall Tau-b statistic (https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Tau-b)
        """

        x, _ = self._preprocessing(ts)
        x, _ = self._drop_missing_values(x)

        anchor_date = ts.index[-1]

        mk_result = mk.seasonal_test(x, period)
        trend, p, Tau = mk_result.trend, mk_result.p, mk_result.Tau
        trend = self._apply_threshold(trend, Tau)

        return anchor_date, trend, p, Tau

    def multivariate_SMKtest(
        self, ts: pd.DataFrame
    ) -> Tuple[datetime, str, float, Dict]:
        """Performs the Multivariate Seasonal Mann-Kendall (SMK) test.

        Args:
            ts: the dataframe of input data with time as index.

        Returns:
            (tuple): tuple containing:

                anchor_date(datetime): the last time point in ts; the date for
                    which alert is triggered.
                trend:_dict: tells the trend (decreasing, increasing, or no trend)
                    for each metric.
                p: p-value of the significance test.
                Tau_dict: Dictionary of Kendall Tau-b statistics for each univariate
                    time series, and Tau_dict["overall"] gives the Tau-b statistic
                    for the multivariate time series.
        """

        anchor_date = ts.index[-1]

        Tau_dict = {}  # contains score for individual cluster and overall
        trend_dict = {}  # contains trend for individual cluster and overall

        x, c = self._preprocessing(ts)

        multi_mk_result = mk.multivariate_test(x)
        trend, p, Tau = multi_mk_result.trend, multi_mk_result.p, multi_mk_result.Tau
        trend = self._apply_threshold(trend, Tau)

        Tau_dict["overall"] = Tau
        trend_dict["overall"] = trend

        for i in range(c):
            x_i, n = self._drop_missing_values(x[:, i])
            # individual Tau score and trend
            try:
                mk_result = mk.seasonal_test(x_i)
                trend_i, Tau_i = mk_result.trend, mk_result.Tau
                trend_i = self._apply_threshold(trend_i, Tau_i)
                Tau_dict[ts.columns[i]] = Tau_i
                trend_dict[ts.columns[i]] = trend_i
            except ZeroDivisionError:
                Tau_dict[ts.columns[i]] = None
                trend_dict[ts.columns[i]] = None

        return anchor_date, trend_dict, p, Tau_dict

    def runDetector(self, ts: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Runs SMK test for a time point in the input data.

        Args:
            ts: the dataframe of input data with noise and seasonality removed.
                Its index is time.

        Returns:
            A dictionary consisting of SMK test statistics for the anchor time
                point, including trend, p-value and Kendall Tau.
        """

        # run SMK test
        if self.multivariate:
            anchor_date, trend, p, Tau = self.multivariate_SMKtest(ts)
        else:
            anchor_date, trend, p, Tau = self.SMKtest(ts, period)

        return {"ds": anchor_date, "trend_direction": trend, "p": p, "Tau": Tau}

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(
        self,
        window_size: int = 20,
        training_days: Optional[int] = None,
        direction: str = "both",
        freq: Optional[str] = None,
        period: Optional[int] = None
    ) -> Sequence[SMKChangePoint]:
        """Runs SMK test sequentially.

        It finds the trend and calculates the related statistics for all time
        points in a given time series.

        Args:
            window_size: int, the number of look back days for checking trend
                persistence
            training_days: int, the number of days for time series smoothing;
                should be greater or equal to window_size
                If training_days is None, we will perform trend detection on the
                whole time series; otherwise, we will perform trend detection
                only for the anchor point using the previous training_days data.
            direction: string, the direction of the trend to be detected, choose
                from {"down", "up", "both"}
            freq: str, the type of seasonality shown in the time series,
                choose from {'weekly','monthly','yearly'}
        """

        self.window_size = window_size
        self.training_days = training_days
        self.direction = direction
        self.freq = freq
        self.period = period

        ts = self.data.to_dataframe().set_index("time")
        ts = ts.dropna(axis=1)
        ts.index = pd.DatetimeIndex(ts.index.values, freq=ts.index.inferred_freq)
        self.ts = ts

        if training_days is None:
            logging.info("Performing trend detection on the whole time series...")
            # check validity of the input value
            if len(ts) < window_size:
                raise ValueError(
                    f"For the whole time series analysis, data must have at "
                    f"least window_size={window_size} points."
                )

        else:
            logging.info(
                f"Performing trend detection for the anchor date {ts.index[-1]}"
                f" with training_days={training_days}..."
            )

            # check validity of the input value
            if training_days < window_size:
                raise ValueError(
                    f"For the anchor date analysis, training days should have "
                    f"at least window_size={window_size} points."
                )

            if len(ts) < training_days:
                raise ValueError(
                    f"For the anchor date analysis, data must have "
                    f"at least training_days={training_days} points."
                )

        # save the trend detection results to dataframe SMK_statistics
        SMK_statistics = pd.DataFrame(columns=["ds", "trend_direction", "p", "Tau"])

        if training_days is not None:  # anchor date analysis for real-time setting
            # only look back training_days for noise and seasonality removal
            start = ts.index[-1] - timedelta(days=training_days)
            ts = ts.loc[start : ts.index[-1], :]
            # deseasonalization
            ts_deseas = self._remove_seasonality(ts, freq=self.freq)
            ts_smoothed = self._smoothing(ts_deseas)  # smoothing
            # append MK statistics to SMK_statistics dataframe
            SMK_statistics = SMK_statistics.append(
                # pyre-ignore[6]: Expected `Union[Dict[Union[int, str], typing.Any], L...
                self.runDetector(ts=ts_smoothed,period=period),
                ignore_index=True,
            )

        else:
            # use the whole time series for for noise and seasonality removal
            ts_deseas = self._remove_seasonality(ts, freq=freq)
            ts_smoothed = self._smoothing(ts_deseas)

            # run detector sequentially with sliding_window for the whole time
            # series
            for t in ts_smoothed.index[window_size:]:
                # look back window_size day for trend detection
                ts_tmp = ts_smoothed.loc[:t, :]
                # append MK statistics to SMK_statistics dataframe
                SMK_statistics = SMK_statistics.append(
                    # pyre-ignore[6]: Expected `Union[Dict[Union[int, str], typing.Any...
                    self.runDetector(ts=ts_tmp, period=period),
                    ignore_index=True,
                )

        self.SMK_statistics = SMK_statistics

        # take the subset for detection with specified trend_direction
        SMK_results = self.get_SMK_results(
            SMK_statistics=SMK_statistics, direction=direction
        )

        return self._convert_detected_tps(SMK_results)

    def get_SMK_results(
        self, SMK_statistics: pd.DataFrame, direction: str
    ) -> pd.DataFrame:
        """Obtain a subset of SMK_statistics given the desired direction"""

        if direction not in ["up", "down", "both"]:
            raise ValueError("direction should be chosen from {'up', 'down', 'both'}")

        if self.multivariate:
            trend_df = pd.DataFrame.from_dict(list(SMK_statistics.trend_direction))
            overall_trend = trend_df["overall"]

            if direction == "down":
                SMK_results = SMK_statistics.loc[overall_trend == "decreasing", :]
            elif direction == "up":
                SMK_results = SMK_statistics.loc[overall_trend == "increasing", :]
            elif direction == "both":
                SMK_results = SMK_statistics.loc[overall_trend != "no trend", :]
        else:
            if direction == "down":
                SMK_results = SMK_statistics.loc[
                    SMK_statistics["trend_direction"] == "decreasing", :
                ]
            elif direction == "up":
                SMK_results = SMK_statistics.loc[
                    SMK_statistics["trend_direction"] == "increasing", :
                ]
            elif direction == "both":
                SMK_results = SMK_statistics.loc[
                    SMK_statistics["trend_direction"] != "no trend", :
                ]
        # pyre-fixme[61]: `SMK_results` may not be initialized here.
        return SMK_results

    def _convert_detected_tps(self, SMK_results: pd.DataFrame) -> List[SMKChangePoint]:
        """Convert the dataframe of detected_tps and Tau into desired format."""

        converted = []

        for _index, row in SMK_results.iterrows():
            t = row["ds"]
            detected_time_point = SMKChangePoint(
                start_time=t,
                end_time=t,
                confidence=1 - row["p"],
                is_multivariate=self.multivariate,
                trend_direction=row["trend_direction"],
                Tau=row["Tau"],
            )
            converted.append(detected_time_point)

        return converted

    def get_SMK_statistics(self) -> pd.DataFrame:
        """Get the dataframe of SMK_statistics."""
        SMK_statistics = self.SMK_statistics
        if SMK_statistics is None:
            raise ValueError("Call detector() first.")
        return SMK_statistics

    def get_top_k_metrics(
        self, time_point: datetime, top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """Get k metrics that show the most significant trend at a time point.

        Works only for multivariate data.

        Args:
            time_point: the time point to be investigated.
            top_k: the number of top metrics.

        Returns:
            a dataframe consists of top_k metrics and their corresponding
                Kendall Tau and trend.
        """

        Tau_df, trend_df = self._metrics_analysis()
        Tau_df = Tau_df.melt(id_vars=["ds"], var_name="metric", value_name="Tau")
        trend_df = trend_df.melt(
            id_vars=["ds"], var_name="metric", value_name="trend_direction"
        )

        if self.training_days is not None:
            time_point = self.data.time.iloc[-1]
            # time_point default to the only anchor date for real-time detection

        # obtain the Tau for all metrics at the time point
        Tau_df_tp = Tau_df.loc[Tau_df["ds"] == time_point, :]
        trend_df_tp = trend_df.loc[trend_df["ds"] == time_point, :]
        SMK_statistics_tp = pd.merge(Tau_df_tp, trend_df_tp)

        # sort the metrics according to their Tau
        if self.direction == "down":
            top_metrics = SMK_statistics_tp.reindex(
                SMK_statistics_tp.Tau.sort_values(axis=0).index
            )
        elif self.direction == "up":
            top_metrics = SMK_statistics_tp.reindex(
                SMK_statistics_tp.Tau.sort_values(axis=0, ascending=False).index
            )
        else:
            assert self.direction == "both"
            top_metrics = SMK_statistics_tp.reindex(
                SMK_statistics_tp.Tau.abs().sort_values(axis=0, ascending=False).index
            )

        if top_k is None:
            # if top_k not specified, return all metrics ranked by Tau
            return top_metrics

        return top_metrics.iloc[:top_k]

    def plot_heat_map(self) -> pd.DataFrame:
        """Plots the Tau of each metric in a heatmap.

        Returns:
            a dataframe contains Tau for all metrics at all time points.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise RuntimeError("requires plotly to be installed")

        Tau_df, _ = self._metrics_analysis()
        Tau_df = Tau_df.set_index("ds")

        fig = go.Figure(
            data=go.Heatmap(
                z=Tau_df.T.values,
                x=Tau_df.index,
                y=Tau_df.columns,
                colorscale="Viridis",
                reversescale=True,
            )
        )

        fig.update_layout(
            xaxis_title="time",
            yaxis_title="value",
            xaxis={"title": "time", "tickangle": 45},
        )

        fig.show()

        return Tau_df

    def _metrics_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.multivariate:
            raise ValueError("Your data is not multivariate.")
        SMK_statistics = self.SMK_statistics
        assert SMK_statistics is not None

        # obtain the Tau for all metrics at all time points
        Tau_df = pd.DataFrame.from_dict(list(SMK_statistics.Tau))
        Tau_df["ds"] = SMK_statistics.ds
        Tau_df = Tau_df.drop(["overall"], axis=1)  # remove overall score

        trend_df = pd.DataFrame.from_dict(list(SMK_statistics.trend_direction))
        trend_df["ds"] = SMK_statistics.ds
        trend_df = trend_df.drop(["overall"], axis=1)  # remove overall trend

        return Tau_df, trend_df

    def plot(
        self,
        detected_time_points: Sequence[SMKChangePoint],
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        changepoint_color: str = "red",
        **kwargs: Any,
    ) -> plt.Axes:
        """Plots the original time series data and the detected time points.

        Args:
            detected_time_points: the detected change points.
            ax: optional matplotlib Axes. If None, creates one.
            figsize: optional figure size. If None, uses (14, 5).
            changepoint_color: the color to use for changepoint vertical lines.
        """
        ts = self.ts
        if ts is None:
            raise ValueError("detector() must be called before plot()")

        with pd.option_context("plotting.matplotlib.register_converters", True):
            if ax is None:
                if figsize is None:
                    figsize = (14, 5)
                _, ax = plt.subplots(figsize=figsize)

            ax.plot(ts.index, ts.values)

            for t in detected_time_points:
                ax.axvline(x=t.start_time, color=changepoint_color)
            else:
                logging.warning("No trend detected!")
            return ax
