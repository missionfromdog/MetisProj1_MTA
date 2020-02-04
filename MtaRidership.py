import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


class MtaRidership:
    """
    A convenience-wrapper for the MTA Project.
    Loads data. Creates Charts.

    Code derived from Alice's Challenge Solutions
    """

    # Class static to simplify and make consistent common columns used as a key
    key_columns = ["C/A", "UNIT", "SCP", "STATION"]


    def __init__(self, mta_week_numbers=None, useFiles=False):
        """
        MtaRidership class constructor

        :param mta_week_numbers:    A list of YYMMDD strings for the turnstile week files to load.
        :type mta_week_numbers:     list
        :param: useFiles:   Flag indicating if the turnstile files should be loaded from
                            the web or the data sub-directory
        :type useFiles:     bool
        :return: MtaRidership
        """

        self.station_ridership = 0

        # Primary DataFrame holding turnstile data
        self.df = pd.DataFrame()

        # DataFrame for turnstile data grouped by Day and counting total daily entries
        self.daily = pd.DataFrame()

        # These are intentionally pointer copies. turnstiles_df and
        # turnstiles_daily are kept for backwards compatibility.
        self.turnstiles_df = self.df
        self.turnstiles_daily = self.daily

        # If no data files are specified, don't try to load or clean them
        if mta_week_numbers is None:
            return

        # Load the turnstile data
        self.get_data(mta_week_numbers,useFiles)

        # Cleanup the data
        self.cleanup()


    def cleanup(self, version=2, addDateTime=False):
        """

        Clean up the loaded data. Two versions of cleanup are available.
        This is the current version. The old version is in the bootstrap function.

        :param version: Select which version of cleanup should be run
        :type version:  bool
        :param addDateTime: Should the DATE_TIME column be created on the self.df DataFrame?
        :type addDateTime: bool
        :return: None
        """

        # Should we call the original bootstrap code?
        if version==1:
            self.bootstrap_original()
            return

        # Remove leading & trailing spaces from column names
        self.formatColumnNames()

        # This seems to take a long time and was only used when
        # visually exploring the data. Put it into an if test so loading it is
        # now configurable.
        if addDateTime:
            self.addDateTimeColumn()

        # The data in these rows is really bad. Occasionally it's ok, but it
        # would be too labor intensive to try and determine which to keep.
        self.dropRowsByColValue('DESC','RECOVR AUD')

        # Create a new pd.DataFrame that contains the data grouped by station
        # and date. Use it to determine how many people passed through a
        # station.
        self.daily= self.makeDaily()


    # Source: http://web.mta.info/developers/turnstile.html
    def get_data(self, week_nums, useFiles=False):
        """
        Load turnstile data from either the MTA web site or downloaded files.
        An example week_nums is [160903, 160910, 160917]

        :param week_nums:   A list of YYMMDD strings for the turnstile week files to load.
        :type week_nums:    list
        :param: useFiles:   Flag indicating if the turnstile files should be loaded from
                            the web or the data sub-directory
        :type useFiles:     bool
        """

        # List of DataFrames, one per file
        dfs = []

        url = "http://web.mta.info/developers/data/nyct/turnstile/turnstile_{}.txt"

        if useFiles:
            url = "./data/turnstile_{}.txt"

        for week_num in week_nums:
            # Replace {} with the element from the list
            file_url = url.format(week_num)

            # Load the CSV into a DataFrame and append it to the list
            dfs.append(pd.read_csv(file_url))

        # Concat all the DataFrames together
        self.df = pd.concat(dfs)


    def format_columns(self):
        """
        Original function to strip spaces from column names and add a DATE_TIME column

        :return: None
        :rtype: None
        """
        self.df.columns = [column.strip() for column in self.df.columns]
        # Take the date and time fields into a single datetime column
        self.df["DATE_TIME"] = pd.to_datetime(self.df.DATE + " " + self.df.TIME,
                                                         format="%m/%d/%Y %H:%M:%S")


    def formatColumnNames(self):
        # Remove leading & tailing space from column names
        self.df.columns = [column.strip() for column in self.df.columns]


    def addDateTimeColumn(self):
        """
        Create a new DATE_TIME column from the date and time fields

        :return: None
        """
        self.df["DATE_TIME"] = pd.to_datetime(self.df.DATE+" "+self.df.TIME,
                                              infer_datetime_format=False)


    def remove_duplicates(self):
        """
        Get rid of the duplicate entry

        :return: None
        """
        self.df.sort_values(MtaRidership.key_columns+["DATE_TIME"], inplace=True, ascending=False)
        self.df.drop_duplicates(subset=MtaRidership.key_columns+["DATE_TIME"], inplace=True)


    def drop_misc_columns(self):
        """
        Drop Exits and Desc columns. To prevent errors in multiple run of cell,
        errors on drop is ignored (e.g. if some columns were dropped already)

        :return: None
        """

        self.df = self.df.drop(["EXITS", "DESC"], axis=1, errors="ignore")


    def group_by_station_and_date(self):
        """
        Populate the daily DataFrame with a new DataFrame built by grouping the df DataFrame
        by the key_columns and DATE. The fist ENTRIES row for each GROUP is selected.

        Also, PREV_DATE and PREV_ENTRIES columns are populated using shift.

        :return: None
        """
        self.daily = self.df.groupby(MtaRidership.key_columns+["DATE"])\
            .ENTRIES.first().reset_index()

        self.daily[["PREV_DATE", "PREV_ENTRIES"]] = (self.daily
                                                                .groupby(MtaRidership.key_columns)[
                                                                    "DATE", "ENTRIES"]
                                                                .transform(lambda grp: grp.shift(1)))
        # transform() takes a function as parameter
        # shift moves the index by the number of periods given (positive or negative)

        # Drop the rows for first date
        self.daily.dropna(subset=["PREV_DATE"], axis=0, inplace=True)
        # axis = 0 means index (=1 means column)


    @staticmethod
    def get_daily_counts(row, max_counter):
        """
        Helper function to calculate ENTRIES-PREV_ENTRIES. The result is tested and
        reversed if negative or set to zero if over max_counter.

        :param row:         A DataFrame with ENTRIES and PREV_ENTRIES columns
        :type row:          pandas.DataFrame
        :param max_counter: The maximum value after which the count is reset to zero
        :type max_counter:  int
        :return:            The calculated difference in values
        :rtype:             int
        """

        counter = row["ENTRIES"] - row["PREV_ENTRIES"]
        if counter < 0:
            counter = -counter
        if counter > max_counter:
            return 0
        return counter


    def count_ridership(self):
        """
        Adds the new column DAILY_ENTRIES to the daily DataFrame. It is populated with the results
        of calling get_daily_counters with a max_counter of 1000000

        :return: None
        """
        # If counter is > 1 million, then the counter might have been reset.
        # Just set it to zero as different counters have different cycle limits

        # apply takes a function as parameter and applies it along the given axis (1=apply by row)
        # apply takes care of passing each row to the function
        self.daily["DAILY_ENTRIES"] = self.daily.apply(self.get_daily_counts, axis=1, max_counter=1000000)


    def n_largest(self, n):
        """
        Convenience method to retrieve the top N MTA stations by traffic. The DataFrame
        created is saved in station_weekly_ridership. It is created using the data
        in the daily DataFrame.

        :param n: Number of highest trafficked stations
        :return: Pandas DataFrame
        """
        station_weekly_ridership = self.daily.groupby(['STATION'])[['DAILY_ENTRIES']].sum()
        return station_weekly_ridership.nlargest(n, 'DAILY_ENTRIES')


    def getMvp(self, N):
        """
        Our Minimum Viable Product. Creates a bar graph with the top
        N highest trafficked MTA stations.

        :param N: Number of top stations (e.g. Top 10, Top 5) to use in the graph
        :type N: int
        :return: Seaborn figure to be shown with plt.show()
        """

        top_N = self.n_largest(N)
        top_N = top_N.div(1000000)
        top_ten_indexed = top_N.reset_index()

        fig = sb.barplot(x=top_ten_indexed['STATION'], y=top_ten_indexed['DAILY_ENTRIES'])
        sb.despine()


        plt.ylabel("People (in millions)", fontsize=10)
        plt.xlabel("Station", fontsize=10, labelpad=10)
        plt.title("Top "+str(N)+" MTA Stations Ridership Volume")
        plt.xticks(fontsize = 8, rotation=80)

        return fig


    def dropRowsByColValue(self, columnName, columnValue):
        """
        Drop rows from the df DataFrame where the entry in columnName equals columnValue

        :param columnName:  a string naming the column
        :type columnName:   str
        :param columnValue: a single object for equality testing
        :type columnValue:  object
        :return:            None
        """
        self.df.drop(index=self.df[self.df[columnName]==columnValue].index,inplace=True)


    def get_hist(self, n):
        """
        Sets up a Matplotlib Histogram that can be displayed with plt.show()

        :param n: Number of MTA Stops to generate histogram
        :type n: int
        :return: None
        """

        plt.hist(self.n_largest(n).DAILY_ENTRIES)
        plt.title("Ridership Distribution - Top 50 MTA Stops")
        plt.xlabel("Ridership Volume")
        plt.ylabel("Number of Stations")


    def iqr(self, N):
        """
        Find Inter-quartile Range of the top N MTA stops

        :param N: The set of largest MTA stops to choose from
        :type N: int
        :return: A DataFrame containing the IQR of the top N values in the data set
        :rtype: pandas.DataFrame
        """

        iqr_df = self.n_largest(N)

        iqr = iqr_df.quantile([.25, .75])

        iqr_sample = iqr_df[iqr_df['DAILY_ENTRIES'] > iqr.DAILY_ENTRIES[0.25]]
        iqr_sample += iqr_df[iqr_df['DAILY_ENTRIES'] < iqr.DAILY_ENTRIES[0.75]]

        return iqr_sample


    def getIqrMvp(self, N):
        """
        Charts the IQR Method

        :param N: Top N MTA stops
        :type: int
        :return: Seaborn chart
        """
        sample = self.iqr(N)

        sample_indexed = sample.reset_index()

        fig = sb.barplot(x=sample_indexed['STATION'], y=sample_indexed['DAILY_ENTRIES'])
        sb.despine()

        plt.ylabel("People")
        plt.xlabel("Station", labelpad=40)
        plt.title("MTA Ridership Volume")
        plt.xticks(rotation=90)

        return fig


    def ridershipSharePercentage(self, topN):
        """
        Calculates the percentage of ridership owned by the top N most trafficked stations

        :param topN: Number of MTA stops
        :type topN: int
        :return: The percentage
        :rtype: float
        """

        top_vol = self.n_largest(topN).sum()[0] #Product

        len(self.daily.groupby('STATION')[['DAILY_ENTRIES']])

        master = self.daily.groupby('STATION')[['DAILY_ENTRIES']].sum()

        bottom_vol = master.nsmallest(len(master)-15, 'DAILY_ENTRIES').sum()[0] # Quotient

        return top_vol / bottom_vol


    def makeDaily(self):
        """
        Creates a new DataFrame with the data grouped by turnstile and day and determine the ridership

        :return: the daily DataFrame
        :rtype: pandas.DataFrame
        """
        # Create a new pd.DataFrame, grouping index columns & DATE
        # - Look at ENTRIES column
        # - Take the first row of this column
        # - Reset the index back to a row number
        self.daily = self.df.groupby(MtaRidership.key_columns + ['DATE']).ENTRIES.first().reset_index()

        # RAW_DAILY_ENTRIES
        # - Group the data by station, then within each group:
        #     - For the ENTRIES column, subtract the current value from the value in the row above
        # - The first row gets NaN since there is no previous row
        self.daily['RAW_DAILY_ENTRIES'] = self.daily.groupby(MtaRidership.key_columns).ENTRIES.diff()

        # Drop the rows in each group with no count (NaN)
        self.daily.dropna(subset=["RAW_DAILY_ENTRIES"], axis=0, inplace=True)

        # Convert the column back to int
        self.daily = self.daily.astype({'RAW_DAILY_ENTRIES': np.int64}, copy=False)

        # DAILY_ENTRIES
        # Some of the counters count down, so take the absolute value
        self.daily['DAILY_ENTRIES'] = np.abs(self.daily['RAW_DAILY_ENTRIES'])

        # Zero-out counts that exceed 2 riders per second every second for 24H
        self.daily.loc[self.daily['DAILY_ENTRIES'] > 24 * 60 * 60 * 2, 'DAILY_ENTRIES'] = 0

        return self.daily


    def bootstrap_original(self):
        """
        Called by __init__. Logic for parsing MTA data.

        :return: None
        """

        self.format_columns()
        self.remove_duplicates()
        self.drop_misc_columns()
        self.group_by_station_and_date()
        self.count_ridership()
        self.station_ridership = self.turnstiles_daily.groupby('STATION')[['DAILY_ENTRIES']].sum()


if __name__ == "__main__":
    mta = MtaRidership([180106],useFiles=True)
    print(mta.n_largest(10))
