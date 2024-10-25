"""
Part 2: Performance Comparisons

**Released: Wednesday, October 16**

In this part, we will explore comparing the performance
of different pipelines.
First, we will set up some helper classes.
Then we will do a few comparisons
between two or more versions of a pipeline
to report which one is faster.
"""

import part1

# importing time
import time 
import matplotlib.pyplot as plt
import pandas as pd


"""
=== Questions 1-5: Throughput and Latency Helpers ===

We will design and fill out two helper classes.

The first is a helper class for throughput (Q1).
The class is created by adding a series of pipelines
(via .add_pipeline(name, size, func))
where name is a title describing the pipeline,
size is the number of elements in the input dataset for the pipeline,
and func is a function that can be run on zero arguments
which runs the pipeline (like def f()).

The second is a similar helper class for latency (Q3).

1. Throughput helper class

Fill in the add_pipeline, eval_throughput, and generate_plot functions below.
"""

# Number of times to run each pipeline in the following results.
# You may modify this if any of your tests are running particularly slow
# or fast (though it should be at least 10).
NUM_RUNS = 10

class ThroughputHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline input sizes
        self.sizes = []

        # Pipeline throughputs
        # This is set to None, but will be set to a list after throughputs
        # are calculated.
        self.throughputs = None

    def add_pipeline(self, name, size, func):
        # adding name , size, func
        self.names.append(name)
        self.sizes.append(size)
        self.pipelines.append(func)



    def compare_throughput(self):
        # Measure the throughput of all pipelines
        # and store it in a list in self.throughputs.
        # Make sure to use the NUM_RUNS variable.
        # Also, return the resulting list of throughputs,
        # in **number of items per second.**

        # pipeline throughputs
        self.throughputs = []

        # access both function and size at same time by using zip
        for func, size in zip(self.pipelines, self.sizes):
            total_duration = 0
            for i in range(NUM_RUNS):
                # getting start and end times of running function, then finding the time to run it
                start = time.time()
                func()
                end = time.time()
                duration = end - start
                # add to total duration
                total_duration += duration
            
            # get average duration and find throughput by dividing size by average time
            avg_duration = total_duration / NUM_RUNS
            throughput = size / avg_duration
            self.throughputs.append(throughput)

        return(self.throughputs)


    def generate_plot(self, filename):
        # Generate a plot for throughput using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        plt.clf() # clearing prev plots in case 
        
        # creating bar plot with names on x axis and throughputs on y axis
        plt.bar(self.names, self.throughputs) 
        plt.xlabel("Pipeline")
        # rotating the xlabels so they don't overlap
        plt.xticks(rotation=45)
        plt.ylabel("Throughput")
        plt.legend(self.names, bbox_to_anchor=(1, 1))
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

"""
As your answer to this part,
return the name of the method you decided to use in
matplotlib.

(Example: "boxplot" or "scatter")
"""

def q1():
    # Return plot method (as a string) from matplotlib
    return('bar')

"""
2. A simple test case

To make sure your monitor is working, test it on a very simple
pipeline that adds up the total of all elements in a list.

We will compare three versions of the pipeline depending on the
input size.
"""

LIST_SMALL = [10] * 100
LIST_MEDIUM = [10] * 100_000
LIST_LARGE = [10] * 100_000_000

def add_list(l):
    sum = 0
    for item in l:
        sum += item
    return(sum)


def q2a():
    # Create a ThroughputHelper object
    h = ThroughputHelper()
    # Add the 3 pipelines.
    # (You will need to create a pipeline for each one.)
    # Pipeline names: small, medium, large
    h.add_pipeline("small", len(LIST_SMALL), lambda: add_list(LIST_SMALL))
    h.add_pipeline( "medium", len(LIST_MEDIUM),lambda: add_list(LIST_MEDIUM))
    h.add_pipeline("large", len(LIST_LARGE), lambda: add_list(LIST_LARGE))
    throughputs = h.compare_throughput()
    # Generate a plot.
    # Save the plot as 'output/q2a.png'.
    h.generate_plot('output/q2a.png')
    
    # Finally, return the throughputs as a list.
    return(throughputs)

"""
2b.
Which pipeline has the highest throughput?
Is this what you expected?

=== ANSWER Q2b BELOW ===

The large list had the highest throughput. I expected this because it was the largest. 

=== END OF Q2b ANSWER ===
"""

"""
3. Latency helper class.

Now we will create a similar helper class for latency.

The helper should assume a pipeline that only has *one* element
in the input dataset.

It should use the NUM_RUNS variable as with throughput.
"""

class LatencyHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline latencies
        # This is set to None, but will be set to a list after latencies
        # are calculated.
        self.latencies = None

    def add_pipeline(self, name, func):
        self.names.append(name)
        self.pipelines.append(func)

    def compare_latency(self):
        # Measure the latency of all pipelines
        # and store it in a list in self.latencies.
        # Also, return the resulting list of latencies,
        # in **milliseconds.**
        # setting latencies to list 
        self.latencies = []
        total_duration = 0
        for func in self.pipelines:
            for i in range(NUM_RUNS):
                start = time.time()
                func()
                end = time.time()
                duration = end - start
                total_duration += duration
            # getting average and converting to milliseconds
            avg_duration = (total_duration/NUM_RUNS) * 1000
            self.latencies.append(avg_duration)
        return(self.latencies)
        

        

    def generate_plot(self, filename):
        # Generate a plot for latency using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        plt.clf() # clearing prev plots in case 
        
        # creating bar plot with names on x axis and throughputs on y axis
        plt.bar(self.names, self.latencies) 
        plt.xlabel("Pipeline")
        # rotating the xlabels so they don't overlap
        plt.xticks(rotation=45)
        plt.ylabel("Latency")
        plt.legend(self.names, bbox_to_anchor=(1, 1))
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        

"""
As your answer to this part,
return the number of input items that each pipeline should
process if the class is used correctly.
"""

def q3():
    # Return the number of input items in each dataset,
    # for the latency helper to run correctly.
    return(1)

"""
4. To make sure your monitor is working, test it on
the simple pipeline from Q2.

For latency, all three pipelines would only process
one item. Therefore instead of using
LIST_SMALL, LIST_MEDIUM, and LIST_LARGE,
for this question run the same pipeline three times
on a single list item.
"""

LIST_SINGLE_ITEM = [10] # Note: a list with only 1 item

def q4a():
    # Create a LatencyHelper object
    h = LatencyHelper()
    # Add the single pipeline three times.
    h.add_pipeline("small", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline( "medium", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("large", lambda: add_list(LIST_SINGLE_ITEM))
    latencies = h.compare_latency()
    # Generate a plot.
    # Save the plot as 'output/q4a.png'.
    h.generate_plot('output/q4a.png')
    # Finally, return the latencies as a list.
    return(latencies)

"""
4b.
How much did the latency vary between the three copies of the pipeline?
Is this more or less than what you expected?

=== ANSWER Q1b BELOW ===

The latencies only vary by 0.0001-0.0004 milliseconds for inputs of the same size, which is expected because this variation is very very small. 

=== END OF Q1b ANSWER ===
"""

"""
Now that we have our helpers, let's do a simple comparison.

NOTE: you may add other helper functions that you may find useful
as you go through this file.

5. Comparison on Part 1

Finally, use the helpers above to calculate the throughput and latency
of the pipeline in part 1.
"""

# You will need these:
part1.load_input
part1.PART_1_PIPELINE

def q5a():
    # Return the throughput of the pipeline in part 1.
    # Create a ThroughputHelper object
    h = ThroughputHelper()
    # getting size of input
    dfs = part1.load_input()
    size = 0
    for df in dfs:
        size += len(df)
    
    # adding pipeline
    h.add_pipeline("PART_1_PIPELINE", size, lambda: part1.PART_1_PIPELINE())
    throughputs = h.compare_throughput()
    # Finally, return the throughputs as a list.
    return(throughputs)
    

def q5b():
    # Return the latency of the pipeline in part 1.
    h = LatencyHelper()
    # getting size of input
    dfs = part1.load_input()
    size = 0
    for df in dfs:
        size += len(df)
    # adding pipeline
    h.add_pipeline("PART_1_PIPELINE", lambda: part1.PART_1_PIPELINE())
    # dividing by size to get actual latency (size = 1)
    latencies = h.compare_latency()
    latencies_adj = [(x / size) for x in latencies]
    return(latencies_adj)

"""
===== Questions 6-10: Performance Comparison 1 =====

For our first performance comparison,
let's look at the cost of getting input from a file, vs. in an existing DataFrame.

6. We will use the same population dataset
that we used in lecture 3.

Load the data using load_input() given the file name.

- Make sure that you clean the data by removing
  continents and world data!
  (World data is listed under OWID_WRL)

Then, set up a simple pipeline that computes summary statistics
for the following:

- *Year over year increase* in population, per country

    (min, median, max, mean, and standard deviation)

How you should compute this:

- For each country, we need the maximum year and the minimum year
in the data. We should divide the population difference
over this time by the length of the time period.

- Make sure you throw out the cases where there is only one year
(if any).

- We should at this point have one data point per country.

- Finally, as your answer, return a list of the:
    min, median, max, mean, and standard deviation
  of the data.

Hints:
You can use the describe() function in Pandas to get these statistics.
You should be able to do something like
df.describe().loc["min"]["colum_name"]

to get a specific value from the describe() function.

You shouldn't use any for loops.
See if you can compute this using Pandas functions only.
"""

def load_input(filename):
    # Return a dataframe containing the population data
    df = pd.read_csv(filename, encoding='latin-1')
    # **Clean the data here**

    # removing world data 
    df = df.drop(df[df.Code == "OWID_WRL"].index)

    # removing continents
    df = df.drop(df[df.Entity == "Africa (UN)"].index)
    df = df.drop(df[df.Entity == "Asia"].index)
    df = df.drop(df[df.Entity == "North America"].index)
    df = df.drop(df[df.Entity == "Europe"].index)
    df = df.drop(df[df.Entity == "South America"].index)
    print(type(df))

    return(df)

    

def population_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
    # getting the years min and max, then dropping countries where min and max year are the same
    year_stats = df.groupby("Entity").agg(min_year=('Year', 'min'), max_year=('Year', 'max')).reset_index()
    year_stats = year_stats[year_stats['min_year'] != year_stats['max_year']]

    # getting the populations of the min and max years, renaming columns to be more clear
    min_population = df.loc[df['Year'] == df.groupby('Entity')['Year'].transform('min'), ['Entity', 'Population (historical)']]
    min_population = min_population.rename(columns={'Population (historical)': 'min_pop'})
    max_population = df.loc[df['Year'] == df.groupby('Entity')['Year'].transform('max'), ['Entity', 'Population (historical)']]
    max_population = max_population.rename(columns={'Population (historical)': 'max_pop'})
    # merging all data together to calculate yearly population diff
    df_merged = year_stats.merge(min_population, on='Entity').merge(max_population, on='Entity')
    # calculating population difference between min and max, the time period, then dividing the difference by the time period
    df_merged['pop_diff'] = df_merged['max_pop'] - df_merged['min_pop']
    df_merged['time_period'] = df_merged['max_year'] - df_merged['min_year']
    df_merged['yearly_pop_diff'] = df_merged['pop_diff'] / df_merged['time_period']

    all_stats = df_merged['yearly_pop_diff'].describe()

    # returning min, median, max, mean, sd

    return([all_stats['min'], all_stats['50%'],all_stats['max'],all_stats['mean'],all_stats['std']])







    

def q6():
    # As your answer to this part,
    # call load_input() and then population_pipeline()
    # Return a list of min, median, max, mean, and standard deviation
    df = load_input('data/population.csv')
    stats = population_pipeline(df)
    return(stats)

"""
7. Varying the input size

Next we want to set up three different datasets of different sizes.s

Create three new files,
    - data/population-small.csv
      with the first 600 rows
    - data/population-medium.csv
      with the first 6000 rows
    - data/population-single-row.csv
      with only the first row
      (for calculating latency)

You can edit the csv file directly to extract the first rows
(remember to also include the header row)
and save a new file.

Make four versions of load input that load your datasets.
(The _large one should use the full population dataset.)
"""

def load_input_small():
    df = pd.read_csv('data/population.csv')
    return(df.head(600))

def load_input_medium():
    df = pd.read_csv('data/population.csv')
    return(df.head(6000))

def load_input_large():
    df = pd.read_csv('data/population.csv')
    return(df)

def load_input_single_row():
    # This is the pipeline we will use for latency.
    df = pd.read_csv('data/population.csv')
    return(df.head(1))

def q7():
    # Don't modify this part
    s = load_input_small()
    m = load_input_medium()
    l = load_input_large()
    x = load_input_single_row()
    return [len(s), len(m), len(l), len(x)]

"""
8.
Create baseline pipelines

First let's create our baseline pipelines.
Create four pipelines,
    baseline_small
    baseline_medium
    baseline_large
    baseline_latency

based on the three datasets above.
Each should call your population_pipeline from Q7.
"""

def baseline_small():
    df = load_input_small()
    return(population_pipeline(df))

def baseline_medium():
    df = load_input_medium()
    return(population_pipeline(df))

def baseline_large():
    df = load_input_large()
    return(population_pipeline(df))

def baseline_latency():
    df = load_input_single_row()
    return(population_pipeline(df))

def q8():
    # Don't modify this part
    _ = baseline_medium()
    return ["baseline_small", "baseline_medium", "baseline_large", "baseline_latency"]

"""
9.
Finally, let's compare whether loading an input from file is faster or slower
than getting it from an existing Pandas dataframe variable.

Create four new dataframes (constant global variables)
directly in the script.
Then use these to write 3 new pipelines:
    fromvar_small
    fromvar_medium
    fromvar_large
    fromvar_latency

As your answer to this part;
a. Generate a plot in output/q9a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, fromvar_small, fromvar_medium, fromvar_large
b. Generate a plot in output/q9b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, fromvar_latency
"""

# writing dataframes directly in script
POPULATION_SMALL = pd.read_csv('data/population.csv').head(600)
POPULATION_MEDIUM = pd.read_csv('data/population.csv').head(6000)
POPULATION_LARGE = pd.read_csv('data/population.csv')
POPULATION_SINGLE_ROW = pd.read_csv('data/population.csv').head(1)

def fromvar_small():
    return(population_pipeline(POPULATION_SMALL))

def fromvar_medium():
    return(population_pipeline(POPULATION_MEDIUM))

def fromvar_large():
    return(population_pipeline(POPULATION_LARGE))

def fromvar_latency():
    return(population_pipeline(POPULATION_SINGLE_ROW))

def q9a():
    # Add all 6 pipelines for a throughput comparison
    # Generate plot in output/q9a.png
    # Return list of 6 throughputs
    h = ThroughputHelper()
    # adding 6 pipelines 
    h.add_pipeline("baseline_small", len(POPULATION_SMALL), baseline_small)
    h.add_pipeline( "baseline_medium", len(POPULATION_MEDIUM),baseline_medium)
    h.add_pipeline("baseline_large", len(POPULATION_LARGE), baseline_large)
    h.add_pipeline("fromvar_small", len(POPULATION_SMALL), fromvar_small)
    h.add_pipeline( "from_varmedium", len(POPULATION_MEDIUM),fromvar_medium)
    h.add_pipeline("fromvar_large", len(POPULATION_LARGE), fromvar_large)
    throughputs = h.compare_throughput()
    # Generate a plot.
    h.generate_plot('output/q9a.png')
    
    # return throughputs
    return(throughputs)
    

def q9b():
    # Add 2 pipelines for a latency comparison
    # Generate plot in ouptut/q9b.png
    # Return list of 2 latencies
    h = LatencyHelper()
    h.add_pipeline("fromvar", fromvar_latency)
    h.add_pipeline("baseline", baseline_latency)
    latencies = h.compare_latency()
    # Generate a plot.
    # Save the plot as 'output/q4a.png'.
    h.generate_plot('output/q9b.png')
    # Finally, return the latencies as a list.
    return(latencies)

"""
10.
Comment on the plots above!
How dramatic is the difference between the two pipelines?
Which differs more, throughput or latency?
What does this experiment show?

===== ANSWER Q10 BELOW =====

The difference in baseline and fromvar differ quite significantly.
The fromvar pipeline had a higher throughput and lower latency.
It looks like throughput and latency differ to a similar degree. However, it appears that throughput differs slightly more.
This experiment showed that larger data had a higher throughput, and that accessing the data directly rather
than calling a function resulted in a lower latency.

===== END OF Q10 ANSWER =====
"""

"""
===== Questions 11-14: Performance Comparison 2 =====

Our second performance comparison will explore vectorization.

Operations in Pandas use Numpy arrays and vectorization to enable
fast operations.
In particular, they are often much faster than using for loops.

Let's explore whether this is true!

11.
First, we need to set up our pipelines for comparison as before.

We already have the baseline pipelines from Q8,
so let's just set up a comparison pipeline
which uses a for loop to calculate the same statistics.

Create a new pipeline:
- Iterate through the dataframe entries. You can assume they are sorted.
- Manually compute the minimum and maximum year for each country.
- Add all of these to a Python list. Then manually compute the summary
  statistics for the list (min, median, max, mean, and standard deviation).
"""

def for_loop_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
    
    values = []
    country = df["Entity"][0]
    min_year_index = 0
    max_year_index = 0
    min_year = float('inf')
    max_year = float('-inf')
    # for every row
    for i in range(len(df)):
        # when country changes, calculate the yearly population diff, reset values
        if country != df["Entity"].iloc[i]:
            if max_year != min_year:
                time_period = max_year - min_year
                population_diff = df["Population (historical)"].iloc[max_year_index] - df["Population (historical)"].iloc[min_year_index]
                values.append(population_diff/time_period)
            country = df["Entity"].iloc[i]
            min_year = float('inf')
            max_year = float('-inf')
        # special case for latency
        if len(df) == 1:
            values.append(df["Population (historical)"].iloc[0])

        # finding min and max
        if df["Year"].iloc[i] < min_year:
            min_year = df["Year"].iloc[i]
            min_year_index = i
        elif df["Year"].iloc[i] > max_year:
            max_year = df["Year"].iloc[i]
            max_year_index = i

    # sorting values to manjally calculate stats
    years_sorted = sorted(values)
    l = len(values)
    year_min = years_sorted[0]
    year_max = years_sorted[-1]
    # finding median under odd and even lengths
    if l % 2 == 1:
        year_median = years_sorted[l // 2]
    else:
        year_median = (years_sorted[(l // 2)-1] + years_sorted[l // 2]) / 2
    year_mean = sum(years_sorted) / l
    year_std = (sum((x - year_mean) ** 2 for x in years_sorted) / l) ** 0.5


    return([year_min, year_median, year_max, year_mean, year_std])




def q11():
    # As your answer to this part, call load_input() and then
    # for_loop_pipeline() to return the 5 numbers.
    # (these should match the numbers you got in Q6.)
    df = load_input('data/population.csv')
    stats = for_loop_pipeline(df)
    return(stats)


"""
12.
Now, let's create our pipelines for comparison.

As before, write 4 pipelines based on the datasets from Q7.
"""

def for_loop_small():
    df = load_input_small()
    return(for_loop_pipeline(df))

def for_loop_medium():
    df = load_input_medium()
    return(for_loop_pipeline(df))

def for_loop_large():
    df = load_input_large()
    return(for_loop_pipeline(df))

def for_loop_latency():
    df = load_input_single_row()
    return(for_loop_pipeline(df))

def q12():
    # Don't modify this part
    _ = for_loop_medium()
    return ["for_loop_small", "for_loop_medium", "for_loop_large", "for_loop_latency"]

"""
13.
Finally, let's compare our two pipelines,
as we did in Q9.

a. Generate a plot in output/q13a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, for_loop_small, for_loop_medium, for_loop_large

b. Generate a plot in output/q13b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, for_loop_latency
"""

def q13a():
    # Add all 6 pipelines for a throughput comparison
    # Generate plot in ouptut/q13a.png
    # Return list of 6 throughputs
    h = ThroughputHelper()
    # adding 6 pipelines 
    h.add_pipeline("baseline_small", len(POPULATION_SMALL), baseline_small)
    h.add_pipeline( "baseline_medium", len(POPULATION_MEDIUM),baseline_medium)
    h.add_pipeline("baseline_large", len(POPULATION_LARGE), baseline_large)
    h.add_pipeline("for_loop_small", len(POPULATION_SMALL), for_loop_small)
    h.add_pipeline( "for_loop_medium", len(POPULATION_MEDIUM),for_loop_medium)
    h.add_pipeline("for_loop_large", len(POPULATION_LARGE), for_loop_large)
    throughputs = h.compare_throughput()
    # Generate a plot.
    h.generate_plot('output/q13a.png')
    
    # return throughputs
    return(throughputs)
    

def q13b():
    # Add 2 pipelines for a latency comparison
    # Generate plot in ouptut/q13b.png
    # Return list of 2 latencies
    h = LatencyHelper()
    h.add_pipeline("for_loop", for_loop_latency)
    h.add_pipeline("baseline", baseline_latency)
    latencies = h.compare_latency()
    # Generate a plot.
    h.generate_plot('output/q13b.png')
    # Finally, return the latencies as a list.
    return(latencies)
    

"""
14.
Comment on the results you got!

14a. Which pipelines is faster in terms of throughput?

===== ANSWER Q14a BELOW =====

The for loop pipeline is faster because it has a higher throughput.

===== END OF Q14a ANSWER =====

14b. Which pipeline is faster in terms of latency?

===== ANSWER Q14b BELOW =====

The for loop pipeline is faster because it has a lower latency.

===== END OF Q14b ANSWER =====

14c. Do you notice any other interesting observations?
What does this experiment show?

===== ANSWER Q14c BELOW =====

I find it really interesting that the for loop pipeline was actually faster. 
This experiment shows that at least for data of the sizes tested, a for loop is faster than vectorized methods. 

===== END OF Q14c ANSWER =====
"""

"""
===== Questions 15-17: Reflection Questions =====
15.

Take a look at all your pipelines above.
Which factor that we tested (file vs. variable, vectorized vs. for loop)
had the biggest impact on performance?

===== ANSWER Q15 BELOW =====

It appears that file vs. variable had a larger impact on latency and throughput, meaning it had the biggest impact on performance overall.

===== END OF Q15 ANSWER =====

16.
Based on all of your plots, form a hypothesis as to how throughput
varies with the size of the input dataset.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q16 BELOW =====

I hypothesize that throughput increases with input dataset size because the overhead for calling the functions is smaller. 

===== END OF Q16 ANSWER =====

17.
Based on all of your plots, form a hypothesis as to how
throughput is related to latency.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q17 BELOW =====

I hypothesize that throughput and latency are inversely correlated but do not have a linear relationship.

===== END OF Q17 ANSWER =====
"""

"""
===== Extra Credit =====

This part is optional.

Use your pipeline to compare something else!

Here are some ideas for what to try:
- the cost of random sampling vs. the cost of getting rows from the
  DataFrame manually
- the cost of cloning a DataFrame
- the cost of sorting a DataFrame prior to doing a computation
- the cost of using different encodings (like one-hot encoding)
  and encodings for null values
- the cost of querying via Pandas methods vs querying via SQL
  For this part: you would want to use something like
  pandasql that can run SQL queries on Pandas data frames. See:
  https://stackoverflow.com/a/45866311/2038713

As your answer to this part,
as before, return
a. the list of 6 throughputs
and
b. the list of 2 latencies.

and generate plots for each of these in the following files:
    output/extra_credit_a.png
    output/extra_credit_b.png
"""

# Extra credit (optional) 

# function for testing cost of cloning dataframe
# one function for using copy and one for using Dataframe(df)

def df_copy(df):
    new_df = df.copy()
    return(new_df)

# functions for copy pipeline

def df_copy_small():
    df = load_input_small()
    return(df_copy(df))

def df_copy_medium():
    df = load_input_medium()
    return(df_copy(df))

def df_copy_large():
    df = load_input_large()
    return(df_copy(df))

def df_copy_latency():
    df = load_input_single_row()
    return(df_copy(df))

# cloning dataframe through constructing a new one from the original

def df_construct(df):
    new_df = pd.DataFrame(df)
    return(new_df)

def df_construct_small():
    df = load_input_small()
    return(df_construct(df))

def df_construct_medium():
    df = load_input_medium()
    return(df_construct(df))

def df_construct_large():
    df = load_input_large()
    return(df_construct(df))

def df_construct_latency():
    df = load_input_single_row()
    return(df_construct(df))

def extra_credit_a():
    # creating pipelines
    h = ThroughputHelper()
    # adding 6 pipelines 
    h.add_pipeline("df_copy_small", len(POPULATION_SMALL), df_copy_small)
    h.add_pipeline( "df_copy_medium", len(POPULATION_MEDIUM),df_copy_medium)
    h.add_pipeline("df_copy_large", len(POPULATION_LARGE), df_copy_large)
    h.add_pipeline("df_construct_small", len(POPULATION_SMALL), df_construct_small)
    h.add_pipeline( "df_construct_medium", len(POPULATION_MEDIUM),df_construct_medium)
    h.add_pipeline("df_construct_large", len(POPULATION_LARGE), df_construct_large)
    throughputs = h.compare_throughput()
    # Generate a plot.
    h.generate_plot('output/extra_credit_a.png')
    
    # return throughputs
    return(throughputs)

def extra_credit_b():
    # creating pipelines
    h = LatencyHelper()
    h.add_pipeline("copy", df_copy_latency)
    h.add_pipeline("construct", df_construct_latency)
    latencies = h.compare_latency()
    # Generate a plot.
    h.generate_plot('output/extra_credit_b.png')
    # Finally, return the latencies as a list.
    return(latencies)

"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part2-answers.txt"
UNFINISHED = 0

def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},{answer}\n')
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Not Implemented\n')
        global UNFINISHED
        UNFINISHED += 1

def PART_2_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    # Q1-5
    log_answer("q1", q1)
    log_answer("q2a", q2a)
    # 2b: commentary
    log_answer("q3", q3)
    log_answer("q4a", q4a)
    # 4b: commentary
    log_answer("q5a", q5a)
    log_answer("q5b", q5b)

    # Q6-10
    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    log_answer("q9a", q9a)
    log_answer("q9b", q9b)
    # 10: commentary

    # Q11-14
    log_answer("q11", q11)
    log_answer("q12", q12)
    log_answer("q13a", q13a)
    log_answer("q13b", q13b)
    # 14: commentary

    # 15-17: reflection
    # 15: commentary
    # 16: commentary
    # 17: commentary

    # Extra credit
    log_answer("extra credit (a)", extra_credit_a)
    log_answer("extra credit (b)", extra_credit_b)

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
=== END OF PART 2 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 2", PART_2_PIPELINE)
