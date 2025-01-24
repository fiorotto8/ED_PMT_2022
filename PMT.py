import numpy as np
import ROOT

def hist(data, x_name, channels=100, linecolor=4, linewidth=4,write=True):
    # Convert list directly to numpy array to avoid redundant loop
    array = np.array(data, dtype="d")
    # Create histogram
    hist = ROOT.TH1D(x_name, x_name, channels, 0.99*np.min(array), 1.01*np.max(array))
    # Use numpy vectorization to fill histogram
    for x in array:
        hist.Fill(x)
    # Set visual attributes and axis titles
    hist.SetLineColor(linecolor)
    hist.SetLineWidth(linewidth)
    hist.GetXaxis().SetTitle(x_name)
    hist.GetYaxis().SetTitle("Entries")
    # Set maximum digits on axes to manage display
    hist.GetYaxis().SetMaxDigits(3)
    hist.GetXaxis().SetMaxDigits(3)
    if write:
        hist.Write()
    return hist
def grapherr(x,y,ex,ey,x_string, y_string,name=None, color=4, markerstyle=22, markersize=2,write=True):
    plot = ROOT.TGraphErrors(len(x),  np.array(x  ,dtype="d")  ,   np.array(y  ,dtype="d") , np.array(   ex   ,dtype="d"),np.array( ey   ,dtype="d"))
    if name is None: plot.SetNameTitle(y_string+" vs "+x_string,y_string+" vs "+x_string)
    else: plot.SetNameTitle(name, name)
    plot.GetXaxis().SetTitle(x_string)
    plot.GetYaxis().SetTitle(y_string)
    plot.SetMarkerColor(color)#blue
    plot.SetMarkerStyle(markerstyle)
    plot.SetMarkerSize(markersize)
    if write==True: plot.Write()
    return plot

class PMTwave:
    def __init__(self, filename):
        """
        Initialize the LecroyWaveform by reading the time and amplitude
        data from the given filename.
        """
        x_list = []
        y_list = []
        self.filename = filename

        with open(filename, 'r') as f:
            found_data_header = False

            for line in f:
                # Strip whitespace around the line
                line = line.strip()

                # Identify when we reach the header 'Time    Ampl'
                if line.startswith("Time"):
                    found_data_header = True
                    continue

                if found_data_header and line:
                    # Each data line should have two columns: time, amplitude
                    parts = line.split()
                    if len(parts) == 2:
                        # Convert them to floats
                        t_val = float(parts[0])
                        a_val = float(parts[1])
                        x_list.append(t_val)
                        y_list.append(a_val)

        # Convert to NumPy arrays
        self.x = np.array(x_list, dtype=np.float64)
        self.y = np.array(y_list, dtype=np.float64)

        self.min_amplitude = self.y.min()
        self.peak_time = self.x[np.argmin(self.y)]
        self.peak_index = np.argmin(self.y)

        self.noisemean, self.noisestd = self.get_stats_first_half()

        self.filtered_y = self.moving_average(200)

        self.minPeakTime, self.maxPeakTime = self.get_peak_times(original=False)

    def get_stats_first_half(self):
        """
        Calculate the mean and standard deviation of the y values in
        the *first half* of the time range spanning from x.min()
        up to self.mintime.

        That "first half" is defined as:
        [x.min(), x.min() + 0.5 * (peaktime - x.min())]
        """
        x_min = self.x.min()

        # Midpoint between x.min() and self.mintime
        x_cutoff = x_min + 0.5 * (self.peak_time - x_min)

        # Select y values whose x is between x_min and x_cutoff
        mask = (self.x >= x_min) & (self.x <= x_cutoff)
        y_segment = self.y[mask]

        # Compute mean and std
        mean_y = np.mean(y_segment)
        std_y = np.std(y_segment)

        return mean_y, std_y

    def get_peak_times(self, fraction=0.1, baseline=None, original=True):
        """
        Find the start time and end time of the negative pulse using a
        fractional threshold approach.

        Parameters
        ----------
        fraction : float
            Fraction of the amplitude difference (baseline - minimum) 
            at which to define the start/end times.

            For example, fraction=0.1 means 10% of the difference between 
            baseline and the negative peak amplitude.

        baseline : float, optional
            If provided, use this as the baseline. Otherwise, the baseline 
            is estimated as the average amplitude of the first few samples.

        Returns
        -------
        (start_time, end_time) : tuple of floats or (None, None) if not found
        """

        # If baseline not provided, estimate from first few samples
        # (You can change the slice size as appropriate)
        if baseline is None:
            baseline = np.mean(self.y[:50])

        # Identify the negative peak amplitude
        peak_amp = self.y[self.peak_index]

        # The threshold is baseline minus "fraction" of the difference
        # between baseline and the negative peak
        #
        # Because the peak is negative, baseline > peak_amp
        # threshold = baseline - fraction*(baseline - peak_amp)
        threshold = baseline - fraction * (baseline - peak_amp)

        if original:
            y_touse = self.y
        else:
            y_touse = self.filtered_y

        # -------------------------------
        # 1) Find the "start time"
        #    Scan from the left up to the peak index to find
        #    when the waveform first goes below threshold
        # -------------------------------
        start_idx = None
        # We go up to peak_index-1 to avoid out-of-range on i+1
        for i in range(self.peak_index):
            if (y_touse[i] > threshold) and (y_touse[i+1] <= threshold):
                start_idx = i
                break

        # -------------------------------
        # 2) Find the "end time"
        #    Scan from the peak index to the right to find
        #    when the waveform goes back above threshold
        # -------------------------------
        end_idx = None
        for j in range(self.peak_index, len(y_touse) - 1):
            if (y_touse[j] <= threshold) and (y_touse[j+1] > threshold):
                end_idx = j
                break

        # Convert indices to times. If either crossing wasn't found,
        # we'll return None for that one.
        start_time = self.x[start_idx] if start_idx is not None else None
        end_time   = self.x[end_idx]   if end_idx   is not None else None

        return start_time, end_time
    
    def moving_average(self, window_size=5):
        """
        Apply a simple moving-average (boxcar) filter to self.y.

        Parameters
        ----------
        window_size : int
            The number of points to include in each moving average window.

        Returns
        -------
        filtered_y : np.ndarray
            The filtered waveform array, the same length as self.y.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        # Create the convolution kernel
        kernel = np.ones(window_size) / window_size

        # 'same' mode returns the same length as self.y
        filtered_y = np.convolve(self.y, kernel, mode='same')

        return filtered_y

    def plot_zoomed_waveform(self, fraction=0.1, baseline=None):
        """
        Plot the waveform zoomed around the start and end of the peak signal.
        Draw two red lines to highlight the start and end times.

        Parameters
        ----------
        fraction : float
            Fraction of the amplitude difference (baseline - minimum) 
            at which to define the start/end times.

        baseline : float, optional
            If provided, use this as the baseline. Otherwise, the baseline 
            is estimated as the average amplitude of the first few samples.
        """

        # Get the start and end times of the peak
        start_time, end_time = self.get_peak_times(fraction, baseline)

        if start_time is None or end_time is None:
            print("Could not determine start or end time of the peak.")
            return

        # Create TGraph directly from NumPy arrays (PyROOT can handle this)
        graph = ROOT.TGraph(self.x.size, self.x, self.y)
        graph.SetTitle("Zoomed LeCroy Waveform;Time (s);Amplitude (V)")

        # Style
        graph.SetLineColor(ROOT.kBlue)
        graph.SetLineWidth(2)
        graph.GetXaxis().SetRangeUser(start_time - 3*(end_time-start_time), end_time + 3*(end_time-start_time))

        # Create a canvas and draw
        canvas = ROOT.TCanvas("ZoomedWaveformCanvas", "Zoomed Waveform Canvas", 1000,1000)
        graph.Draw("ALP")

        # Draw red lines for start and end times
        line_start = ROOT.TLine(start_time, self.y.min(), start_time, self.y.max())
        line_start.SetLineColor(ROOT.kRed)
        line_start.SetLineStyle(1)
        line_start.SetLineWidth(3)
        line_start.Draw()

        line_end = ROOT.TLine(end_time, self.y.min(), end_time, self.y.max())
        line_end.SetLineColor(ROOT.kRed)
        line_end.SetLineStyle(1)
        line_end.SetLineWidth(3)
        line_end.Draw()

        # Set axis ranges to zoom around the peak
        canvas.SetGrid()
        canvas.Update()
        canvas.Modified()
        canvas.GetFrame().SetBorderSize(12)
        canvas.SetLeftMargin(0.15)
        canvas.SetBottomMargin(0.15)
        canvas.SetRightMargin(0.15)
        canvas.SetTopMargin(0.15)
        canvas.Update()

        # Keep the canvas open until user presses Enter in the terminal
        input("Press <ENTER> to close the canvas...")

    def plot_zoomed_waveform_with_filtered(self, fraction=0.1, baseline=None, window_size=200):
        """
        Plot the original waveform and a filtered waveform, both zoomed
        around the start and end of the negative peak. Red lines highlight
        the start and end times determined from the original waveform, while
        green lines highlight those from the filtered waveform.

        Parameters
        ----------
        fraction : float
            Fraction of the amplitude difference (baseline - minimum) 
            at which to define the start/end times (used by get_peak_times).

        baseline : float, optional
            If provided, use this as the baseline. Otherwise, the baseline 
            is estimated internally (in get_peak_times).

        window_size : int
            Window size for the moving-average (boxcar) filter.
        """
        import ROOT  # Ensure PyROOT is available
        
        # 1) Get peak times from the original waveform
        start_time_orig, end_time_orig = self.get_peak_times(fraction, baseline)
        if start_time_orig is None or end_time_orig is None:
            print("Could not determine start or end time of the peak for the original waveform.")
            start_time_filt, end_time_filt = 0, 0

        # 3) Get peak times from the filtered waveform
        start_time_filt, end_time_filt = self.get_peak_times(fraction, baseline=self.filtered_y[:50].mean(),original=False)
        if start_time_filt is None or end_time_filt is None:
            print("Could not determine start or end time of the peak for the filtered waveform.")
            start_time_filt, end_time_filt = 0, 0

        # 4) Create TGraph for the original data
        graph_orig = ROOT.TGraph(self.x.size, self.x, self.y)
        graph_orig.SetTitle("Zoomed Waveform (Original + Filtered);Time (s);Amplitude (V)")
        graph_orig.SetLineColor(ROOT.kBlue)
        graph_orig.SetLineWidth(2)

        # 5) Create TGraph for the filtered data
        graph_filt = ROOT.TGraph(self.x.size, self.x, self.filtered_y)
        graph_filt.SetLineColor(ROOT.kOrange + 1)
        graph_filt.SetLineWidth(2)

        # 6) Compute a suitable zoom window around the peak
        time_span = end_time_orig - start_time_orig
        x_min_zoom = start_time_orig - 2 * time_span
        x_max_zoom = end_time_orig + 2 * time_span

        # 7) Create a canvas and draw the original waveform
        canvas = ROOT.TCanvas("ZoomedWaveformCanvas", "Zoomed Waveform Canvas", 1000,1000)
        graph_orig.GetXaxis().SetRangeUser(-0.2E-6, 0.2E-6)
        graph_orig.Draw("AL")  # Draw axes and line

        # 8) Overdraw the filtered waveform on the same canvas
        graph_filt.Draw("L SAME")  # 'L' for line; 'SAME' to overlay

        # 9) Draw red lines for the start/end times (original waveform)
        y_min = min(self.y.min(), self.filtered_y.min())
        y_max = max(self.y.max(), self.filtered_y.max())
        line_start_orig = ROOT.TLine(start_time_orig, y_min, start_time_orig, y_max)
        line_start_orig.SetLineColor(ROOT.kRed)
        line_start_orig.SetLineStyle(1)
        line_start_orig.SetLineWidth(3)
        line_start_orig.Draw()

        line_end_orig = ROOT.TLine(end_time_orig, y_min, end_time_orig, y_max)
        line_end_orig.SetLineColor(ROOT.kRed)
        line_end_orig.SetLineStyle(1)
        line_end_orig.SetLineWidth(3)
        line_end_orig.Draw()

        # 10) Draw green lines for the start/end times (filtered waveform)
        line_start_filt = ROOT.TLine(start_time_filt, y_min, start_time_filt, y_max)
        line_start_filt.SetLineColor(ROOT.kGreen + 2)
        line_start_filt.SetLineStyle(2)
        line_start_filt.SetLineWidth(3)
        line_start_filt.Draw()

        line_end_filt = ROOT.TLine(end_time_filt, y_min, end_time_filt, y_max)
        line_end_filt.SetLineColor(ROOT.kGreen + 2)
        line_end_filt.SetLineStyle(2)
        line_end_filt.SetLineWidth(3)
        line_end_filt.Draw()

        # 11) Final canvas settings
        canvas.SetGrid()
        canvas.Update()
        canvas.Modified()
        canvas.GetFrame().SetBorderSize(12)
        canvas.SetLeftMargin(0.15)
        canvas.SetBottomMargin(0.15)
        canvas.SetRightMargin(0.15)
        canvas.SetTopMargin(0.15)
        canvas.Update()

        # 12) Keep the canvas open until user presses Enter in the terminal
        #input("Press <ENTER> to close the canvas...")
        # Strip .txt from filename and add .png, change folder to 'picture/'
        output_filename = self.filename.replace('waves/', 'picture/').replace('.txt', '.png')
        canvas.SaveAs(output_filename)

    def plot_waveform(self, original=True, print=True):
        """
        Plot the waveform using ROOT. This method creates a TGraph and
        draws it on a canvas.

        Note: You must run this in an environment where ROOT is set up
        and PyROOT is available.
        """
        if original:
            y_touse = self.y
        else:
            y_touse = self.filtered_y

        # Create TGraph directly from NumPy arrays (PyROOT can handle this)
        graph = ROOT.TGraph(self.x.size, self.x, y_touse)
        graph.SetTitle("LeCroy Waveform;Time (s);Amplitude (V)")

        # Style
        graph.SetLineColor(ROOT.kBlue)
        graph.SetLineWidth(2)

        # Create a canvas and draw
        canvas = ROOT.TCanvas("WaveformCanvas", "Waveform Canvas", 1000,1000)
        graph.Draw("ALP")
        canvas.Update()

        # Keep the canvas open until user presses Enter in the terminal
        if print: input("Press <ENTER> to close the canvas...")
        else: canvas.SaveAs('picture/' + self.filename.replace('waves/', '').replace('.txt', '.png'))