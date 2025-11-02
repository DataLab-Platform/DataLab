# Release notes #

See DataLab [roadmap page](https://datalab-platform.com/en/contributing/roadmap.html) for future and past milestones.

## DataLab Version 1.0.0 ##

💥 New features and enhancements:

* **Annotation management**: New features for managing plot annotations on signals and images
  * New "Annotations" submenu in Edit menu with dedicated annotation operations
  * **Copy annotations**: Copy annotations from selected object to clipboard
    * Only enabled when exactly one object with annotations is selected
  * **Paste annotations**: Paste annotations from clipboard to selected object(s)
    * Merges pasted annotations with any existing annotations on target objects
    * Only enabled when annotations clipboard is not empty
    * Works with multiple selected objects for batch annotation application
  * **Edit annotations**: Open annotation editor dialog for interactive annotation management
    * Create, modify, and delete plot annotations (shapes, labels, markers, etc.)
    * Visual editor with same PlotPy tools as main plot view
    * Changes are immediately applied to selected object
  * **Import annotations**: Load annotations from .dlabann JSON files
    * Merges imported annotations with existing annotations
    * Supports versioned annotation format for future compatibility
  * **Export annotations**: Save annotations to .dlabann JSON files
    * Preserves all annotation properties for later reuse
    * Human-readable JSON format with version information
  * **Delete annotations**: Remove all annotations from selected object(s)
    * Only enabled when at least one selected object has annotations
    * Confirmation dialog prevents accidental deletion
    * Works with multiple selected objects for batch deletion
  * Annotations are managed using Sigima 1.0 annotation API
  * Annotations are stored in object metadata and preserved across save/load operations
  * Clipboard-based workflow enables efficient annotation reuse across objects
  * Independent from ROI management - annotations and ROIs can coexist on same object

* **Metadata management improvements**: Enhanced metadata handling with clipboard-based workflow
  * Reorganized Edit menu with new "Metadata" submenu grouping all metadata operations
  * **Paste metadata** action now only enabled when metadata clipboard is not empty
    * Prevents confusion when no metadata is available to paste
    * Clipboard state is checked dynamically as object selection changes

* **Bug fixes**:
  * **Action enable states not updated after operations**: Fixed issue where menu action states
    were not updated after performing annotation or metadata operations
    * After copying/pasting/deleting annotations or metadata, action states now refresh immediately
    * Users no longer need to select another object to see correct action enabled/disabled states
    * Examples: "Delete annotations" now disables immediately when annotations are deleted,
      "Paste annotations" enables immediately when annotations are copied
    * Improves user experience by keeping UI state synchronized with actual object state

* **Interactive object creation and processing**: New interactive workflow for creating and processing objects
  * **Interactive object creation**: Signal/image creation parameters can now be modified after creation
    * When creating a new signal or image (Gaussian, sine, etc.), a new "Creation" tab appears in the object properties panel
    * Users can modify creation parameters (amplitude, frequency, size, etc.) and click "Apply" to regenerate the object in-place
    * Object is updated without creating a new one, preserving any subsequent processing or analysis
    * Only available for objects created with parameter classes (not for imported data)
    * Useful for exploring different parameter values or fine-tuning object characteristics
  * **Interactive 1-to-1 processing**: Processing parameters can be adjusted and reapplied after initial processing
    * When applying a 1-to-1 processing operation with parameters (Gaussian filter, threshold, etc.), a new "Processing" tab appears
    * Users can modify processing parameters and click "Apply" to re-process the object with updated values
    * Result object is updated in-place with new processing parameters
    * Processing metadata stores: parameters (JSON), source object UUID, and function name
    * Source object validation: displays error message if source object was deleted
    * Only works for processing functions with parameter classes (filtering, morphology, etc.)
    * Not supported for processing without parameters (absolute, inverse, etc.) - these work as before
    * Intentionally not supported for 2-to-1 or n-to-1 processing patterns (use cases don't benefit from interactive behavior)
    * Ideal for iteratively tuning processing parameters while observing results in real-time
  * **Recompute feature**: Quickly reprocess objects with their stored processing parameters
    * New "Recompute" action in Edit menu (keyboard shortcut: Ctrl+R)
    * Allows reprocessing one or multiple objects that have stored 1-to-1 processing parameters
    * When source object data changes, recompute updates the processed object(s) accordingly
    * Works with both individual objects and groups of objects
    * Shows progress dialog for batch recomputing multiple objects
    * Objects without recomputable parameters are automatically skipped
    * Useful for updating results when input data is modified or to propagate changes through a processing chain
  * **Select source objects**: Navigate to source objects used in processing
    * New "Select source objects" action in Edit menu
    * Automatically selects the source object(s) that were used to create the currently selected processed object
    * Works for all processing patterns (1-to-1, 2-to-1, n-to-1) by reading processing metadata
    * Handles multiple source objects (e.g., from n-to-1 operations)
    * Shows informative messages if source objects no longer exist or if some are missing
    * Useful for understanding processing workflows and tracing data lineage
    * Helps users navigate complex processing chains by jumping back to original source data
  * **Processing history**: New "History" tab shows object processing lineage
    * Appears in the Properties panel for all signal and image objects
    * Displays a hierarchical tree showing how the object was created and processed
    * Shows processing chain from oldest (original/created object) to newest (current object)
    * Displays function names for each processing step in a clear, indented format
    * Indicates when source objects have been deleted or when multiple sources were used
    * Automatically updates when object selection changes
    * Useful for understanding complex processing workflows and documenting analysis steps
    * Text is selectable and copyable for documentation purposes

* **Non-uniform coordinate support for images**: Images can now have non-uniform pixel spacing
  * Image objects now support both uniform and non-uniform coordinate systems:
    * Uniform coordinates: regular grid with constant pixel spacing (traditional mode)
    * Non-uniform coordinates: arbitrary X and Y coordinate arrays for each pixel
  * New "Set uniform coordinates" feature in Processing menu > Axis transformation:
    * Converts images with non-uniform coordinates to uniform coordinate system
    * Automatically calculates optimal uniform spacing from non-uniform arrays
    * Useful for standardizing coordinate systems before certain operations
  * Enhanced "Calibration" feature with polynomial support:
    * Now supports polynomial calibration up to cubic order for all axes (X, Y, Z)
    * Polynomial transformation: `output = a0 + a1*input + a2*input² + a3*input³`
    * For X and Y axes: Creates non-uniform coordinates to represent polynomial transformations
    * For Z axis: Applies polynomial transformation to pixel values
    * Linear calibration is achieved by setting quadratic and cubic coefficients to zero
    * Ideal for correcting non-linear distortions in imaging systems or sensors
  * Updated image organization features to handle non-uniform coordinates:
    * "Distribute on grid" now works with both uniform and non-uniform coordinate images
    * "Reset image positions" maintains coordinate system type (uniform/non-uniform)
    * Plot display automatically adapts to show correct pixel positions
  * File I/O support:
    * HDF5 format preserves non-uniform coordinate information
    * Coordinated text files can store and load non-uniform coordinate arrays
  * All image processing operations handle non-uniform coordinates appropriately
  * Performance optimized: Operations on uniform coordinates use fast array indexing when possible

* **DateTime signal support**: Signals can now handle time-series data with datetime X-axis
  * Automatic datetime detection when opening CSV files with timestamp columns
    * DataLab automatically recognizes and converts datetime values (e.g., "2025-06-19 10:00:00") to numeric timestamps
    * Supports common datetime formats from data loggers and measurement instruments
    * Preserves datetime context for proper display and analysis
  * Time units can be configured (seconds, milliseconds, microseconds, minutes, hours)
  * **Automatic datetime axis formatting**: When displaying signals with datetime X-axis, the plot automatically switches to datetime-formatted axis labels
    * X-axis labels show human-readable timestamps (e.g., "10:30:45") instead of numeric values
    * Format adjusts based on signal time unit: sub-second units (ms/μs/ns) show fractional seconds
    * Axis automatically reverts to standard formatting when no datetime signals are displayed
    * Uses signal metadata for custom datetime formats when available
  * **Configurable datetime formats**: Users can customize the datetime axis label format via Settings dialog
    * Separate format strings for standard time units (s/min/h) and sub-second units (ms/μs/ns)
    * Uses Python's strftime format codes for full flexibility
    * Settings are persisted across sessions
    * Default formats: "%H:%M:%S" for standard units, "%H:%M:%S.%f" for sub-second precision
  * Enhanced signal visualization with datetime-aware axis formatting
  * CSV export maintains datetime format for easy data sharing
  * Ideal for analyzing sensor data, temperature logs, and time-series measurements
  * This closes [Issue #258](https://github.com/DataLab-Platform/DataLab/issues/258) - Add support for CSV files with date/time X-axis

* **Enhance metadata handling by adding function name context to results**:
  * When computation functions (e.g., `full_width_at_y`, `x_at_y`) create results with parameters, the function name is now stored
  * Analysis parameters tab displays function name as a comment above parameter information for better context
  * Helps distinguish identical parameter classes (e.g., `OrdinateParam`) used by different functions
  * Improves clarity when reviewing analysis results with multiple computation parameters

* Multi-object property editing:
  * The properties panel now supports applying property changes to multiple selected objects simultaneously.
  * When multiple signals or images are selected, modifying and applying properties updates all selected objects, not just the current one.
  * Only the properties that were actually modified are applied to the selected objects, preserving individual object properties that were not changed.
  * Typical use case: changing the LUT boundaries (Z scale bounds) or colormap of multiple images at once, or adjusting display properties for a group of signals.
  * This significantly improves workflow efficiency when working with multiple objects that need similar property adjustments.

* Result metadata management:
  * **Merged result labels**: All analysis results for a signal or image are now displayed in a single consolidated, read-only label
    * Previously, each analysis result (FWHM, FW1e2, centroid, etc.) created a separate editable label on the plot
    * Now, all results for an object are merged into one read-only label, providing a cleaner and more organized display
    * The merged label automatically updates when new results are computed or existing results are removed
    * Results are visually separated within the label using horizontal dividers for clarity
    * The label is read-only to prevent accidental deletion; individual results can be removed via the Analysis menu
    * Significantly reduces visual clutter when multiple analysis operations are performed on the same object
    * Improves plot readability and makes it easier to review all computed results at a glance
    * The merged label is automatically positioned and managed by DataLab, freeing users from label placement concerns
  * Deleting result labels (e.g., FWHM, statistics) from the plot now permanently removes the associated metadata.
  * Previously, when a result label was deleted from the plot, the underlying result metadata remained in the object, causing the label to reappear when re-selecting the signal or image.
  * Now, when a user deletes a result label via the plot interface, the associated result metadata is automatically removed from the object.
  * This provides more intuitive behavior and gives users better control over which results are kept or discarded.
  * Result deletion is persistent: labels do not reappear when de-selecting and re-selecting the object.
  * This enhancement applies to all result types including geometry results (FWHM, centroid, peak detection) and table results (statistics, pulse analysis).
  * **Individual result deletion**: Results can now be deleted individually from the Analysis menu
    * New "Delete results" submenu in the Analysis menu, similar to the ROI removal feature
    * Lists all individual results for the selected object(s) with their titles
    * Includes a separator and a "Delete all results..." option for bulk deletion
    * When an individual result is selected for deletion, only that specific result is removed from the object
    * Provides more granular control over result management compared to deleting all results at once
    * Useful for removing outdated or incorrect analysis results while keeping others
    * **Fixed**: Result deletion now properly removes all associated metadata including parameter metadata (`__param_str` and `__param_html` entries)

* Internal console status indicator added to the status bar:
  * The status bar now features an indicator for the internal console, visible only when the console is hidden.
  * Clicking the indicator opens the internal console.
  * The icon turns red if an error or warning is logged, alerting the user to check the console.

* Configuration folder versioning for major version coexistence:
  * Configuration folders are now versioned by major version to allow different major versions to coexist on the same machine without interfering with each other.
  * Configuration folder naming:
    * v0.x: `.DataLab` (unchanged for backward compatibility)
    * v1.x: `.DataLab_v1` (new)
    * v2.x: `.DataLab_v2` (future)
  * This enables users to run both v0.20 and v1.0 simultaneously during the migration period, which is particularly useful for plugin developers who need to test their code against both versions.
  * Each version maintains its own isolated configuration, plugins directory, and settings, preventing any conflicts or data loss.

* New common signal/image features:
  * Added two options for signal and image creation:
    * "Use xmin and xmax bounds from current signal when creating a new signal" (default: disabled)
    * "Use dimensions from current image when creating a new image" (default: enabled)
    * If enabled, the new signal/image will use the xmin/xmax bounds or dimensions of the current signal/image, if any
  * **Add metadata**: New feature to add custom metadata items to selected signals or images
    * Accessible via Edit menu → "Add metadata..."
    * Allows adding a metadata key-value pair to one or multiple selected objects
    * Value pattern supports Python format strings with placeholders: `{title}`, `{index}`, `{count}`, `{xlabel}`, `{xunit}`, `{ylabel}`, `{yunit}`, `{metadata[key]}`
    * Supports uppercase/lowercase modifiers (e.g., `{title:upper}`, `{title:lower}`)
    * Automatic type conversion: string, float, integer, or boolean
    * Real-time preview showing how values will be set for each object
    * Useful for tagging objects with experiment IDs, sample names, processing steps, etc.
    * Integrates seamlessly with existing metadata operations (copy, paste, import, export)
  * Added `phase` (argument) feature to extract the phase information from complex signals or images.
    * This complements the "Absolute value" (modulus) feature.
  * Added operation to create complex-valued signal/image from real and imaginary parts.
  * Added operation to create complex-valued signal/image from magnitude and phase (this closes [Issue #216](https://github.com/DataLab-Platform/DataLab/issues/216)).
  * Standard deviation of the selected signals or images (this closes [Issue #196](https://github.com/DataLab-Platform/DataLab/issues/196)):
    * The standard deviation is calculated across the selected signals or images.
    * This is a complement to the "Average" feature.
  * Generate new signal or image: Poisson noise.
  * Add noise to the selected signals or images (this closes [Issue #201](https://github.com/DataLab-Platform/DataLab/issues/201)): Gaussian, Poisson or uniform noise can be added.
  * Save to directory feature (this closes [Issue #227](https://github.com/DataLab-Platform/DataLab/issues/227)).
  * Deconvolution in the frequency domain (this closes [Issue #189](https://github.com/DataLab-Platform/DataLab/issues/189)).
  * Signals to image conversion:
    * Convert collections of signals into 2D images for visualization
    * Select multiple signals and use "Combine signals into image" operation
    * Two orientation modes:
      * As rows: Each signal becomes a row in the resulting image (useful for spectrograms)
      * As columns: Each signal becomes a column in the resulting image (useful for waterfall displays)
    * Optional normalization:
      * Normalize signals before combining using Z-score, Min-Max, or Maximum methods
      * Normalizes each signal independently to handle different amplitude ranges
      * Enables better visualization of signals with varying scales
    * Validates signal compatibility (all must have same size)
    * Resulting image automatically appears in Image panel with appropriate dimensions
    * Supports cross-panel operations: Images created from signals are properly routed to Image panel
    * Group handling: Results maintain group organization when working with signal groups
    * Typical use cases:
      * Creating heatmaps from time-series data
      * Building spectrograms from frequency domain signals
      * Generating waterfall displays for multi-channel data
      * Visualizing signal evolution over time or parameter sweeps

* New ROI (Region of Interest) management features:
  * New "ROI" menu, between "Edit" and "Operations" menus.
  * This menu provides quick access to ROI creation, editing, and deletion tools.
  * New features have also been added to the "ROI" menu:
    * "Create ROI grid": for images, see below.
    * "Copy": copies the selected ROI to the clipboard.
    * "Paste": pastes the ROI from the clipboard to the selected object (signal or image).
    * "Export": exports the selected ROI to a file (JSON format).
    * "Import": imports a ROI from a file (JSON format).
    * Individual ROI removal: new sub-menu "Remove" in the "ROI" menu, to remove ROIs individually instead of removing all ROIs at once.
  * ROI editor now supports setting the ROI title:
    * When adding ROIs interactively, the user can now set a title from the object list on the right side of the ROI editor.
    * After validating the dialog, each ROI title may also be edited in the confirmation dialog.
  * New "Create ROI grid" feature:
    * This feature allows to create a grid of ROIs over an image.
    * The grid is defined using a dialog that allows the user to:
      * Specify the number of rows and columns, as well as the spacing between the ROIs.
      * Import/export ROI grids to/from files (JSON format).
      * Preview the ROI grid before creating it.
  * New "Inverse ROI logic" option for image ROIs:
    * Added support for inverse region selection in image ROIs (rectangles, circles, polygons).
    * When enabled, the ROI represents the area outside the defined shape instead of inside.
    * Available in the ROI editor dialog for all image ROI types.
    * Allows for more flexible region-based analysis and processing operations.
    * Signal ROIs remain unaffected as the inverse concept applies only to 2D regions

* New image operation, processing and analysis features:
  * **Enhanced profile extraction dialog**: Improved user interface for extracting line and average profiles
    * Click directly on the X-profile plot (bottom) to switch to horizontal profile extraction
    * Click directly on the Y-profile plot (right) to switch to vertical profile extraction
    * Direction switching is intuitive and immediate - no need to open parameters dialog
    * Mouse clicks on profile plots no longer select the curve, avoiding visual clutter
    * Works for both "Line profile" and "Average profile" extraction modes
    * Existing "Edit profile parameters" button still available for precise parameter control
    * Significantly improves workflow when comparing horizontal and vertical profiles
    * This closes [Issue #156](https://github.com/DataLab-Platform/DataLab/issues/156) - Image/Average profile: click directly on vertical/horizontal plot to select the profile
  * New 2D ramp image generator.
    * This closes [Issue #203](https://github.com/DataLab-Platform/DataLab/issues/203).
  * Convolution.
  * New "2D resampling" feature:
    * This feature allows to resample 2D images to a new coordinate grid using interpolation.
    * It supports two resampling modes: pixel size and output shape.
    * Multiple interpolation methods are available: linear, cubic, and nearest neighbor.
    * The fill value parameter controls how out-of-bounds pixels are handled, with support for numeric values or NaN.
    * It is implemented in the `sigima.proc.image.resampling` function.
    * This adds comprehensive 2D resampling capabilities to complement existing geometric transformation features.
    * This closes [Issue #208](https://github.com/DataLab-Platform/DataLab/issues/208) - Add support for image resampling.
  * New "Frequency domain Gaussian filter" feature:
    * This feature allows to filter an image in the frequency domain using a Gaussian filter.
    * It is implemented in the `sigima.proc.image.frequency_domain_gaussian_filter` function.
    * This closes [Issue #205](https://github.com/DataLab-Platform/DataLab/issues/205) - Add Gaussian frequency filter for images.
  * New "Erase area" feature:
    * This feature allows to erase an area in an image as defined by a region of interest
    * It is implemented in the `sigima.proc.image.erase` function
    * This closes [Issue #204](https://github.com/DataLab-Platform/DataLab/issues/204) - Add support for background subtraction and inpainting on ROI
  * Improved centroid computation:
    * New adaptive method (`get_centroid_auto`) selects the most reliable centroid estimation (Fourier or fallback) based on internal consistency checks with projected profile methods.
    * Projected profile-based centroid (`get_projected_profile_centroid`) now available for robust, fast, and noise-tolerant estimation — performs well in truncated or asymmetric images.
    * GUI barycenter tool now uses the auto-adaptive method by default for better accuracy and reliability in edge cases.
    * See [Issue #251](https://github.com/DataLab-Platform/DataLab/issues/251) - Inaccurate centroid estimation in edge cases with `get_centroid_fourier`
    * In interactive tools, the centroid is now computed using the `measure.centroid` function from `scikit-image`, which provides a better compromise between speed and accuracy and is suitable for real-time applications.
  * Horizontal and vertical projections.
    * This closes [Issue #209](https://github.com/DataLab-Platform/DataLab/issues/209) - Parameters / Sum of pixels, sum of pixel rows, sum of pixel columns.

* New signal operation, processing and analysis features:
  * New signal generators: linear chirp, logistic function, Planck function.
    * This closes [Issue #213](https://github.com/DataLab-Platform/DataLab/issues/213).
  * **X-array compatibility checking for multi-signal computations**:
    * Added comprehensive x-array compatibility validation for all signal operations that take multiple signals as input (N>1 signals).
    * **Consistent behavior** for all X-array incompatibilities:
      * **Seamless processing**: When signals have identical X arrays, operations proceed normally without interruption.
      * **User choice with interpolation**: When signals have any X-array differences (different sizes or ranges), user is presented with a dialog to either cancel or interpolate signals to match.
      * **No blocking errors**: Operations are never blocked when interpolation can provide a technical solution.
    * **New configuration setting**: Added `xarray_compat_behavior` setting in Processing preferences with options:
      * "Ask user" (default): Shows dialog for user decision when X arrays differ
      * "Interpolate automatically": Automatically interpolates signals when X arrays differ
    * **Enhanced user experience**: Clear dialogs explain compatibility options and always offer interpolation as a solution.
    * **Comprehensive coverage**: Affects all multi-signal operations including:
      * N-to-1 computations (e.g., average of multiple signals, arithmetic operations)
      * 2-to-1 computations in both pairwise and single operand modes (e.g., addition, subtraction, convolution)
    * **Intelligent interpolation**: Uses Sigima's robust interpolation functions to ensure accurate signal matching when requested.
    * This enhancement prevents unexpected results from operations on incompatible signal arrays while providing a consistent and user-friendly approach to handling X-array differences.
  * **Enhanced curve fitting functions**: Significantly improved curve fitting capabilities with advanced parameter estimation:
    * **Intelligent initial parameter estimation**: All curve fitting functions now use advanced algorithms from the Sigima library for automatic initial parameter estimation, resulting in:
      * More robust convergence to optimal solutions
      * Reduced fitting failures and improved success rates
      * Better handling of noisy or challenging experimental data
      * Faster convergence with fewer iterations required
    * **Enhanced fitting algorithms**: Improved implementations for all curve fitting types:
      * Gaussian, Lorentzian, and Voigt profile fitting with better peak detection
      * Exponential and double exponential fitting with enhanced decay/growth parameter estimation
      * Sinusoidal fitting with FFT-based frequency estimation for improved accuracy
      * Planckian (blackbody) fitting with physics-correct parameter bounds
      * Asymmetric peak fitting (two-half Gaussian) with independent left/right parameters
      * CDF (Cumulative Distribution Function) fitting for statistical analysis
    * **Locked parameter support** (requires PlotPy v2.8.0):
      * Added ability to lock individual fit parameters during automatic optimization
      * Users can manually adjust parameters and lock them to prevent changes during auto-fit
      * Visual indicators: locked parameters show 🔒 emoji and are grayed out with disabled controls
      * Enables partial optimization workflows: fix well-determined parameters, optimize uncertain ones
      * All optimization algorithms (simplex, Powell, BFGS, L-BFGS-B, conjugate gradient, least squares) support locked parameters
      * Improves fit convergence by reducing problem dimensionality
      * Lock/unlock parameters via the parameter settings dialog (gear icon)
    * **Improved user experience**: The curve fitting dialogs now provide:
      * More reliable initial parameter guesses reducing manual adjustment needs
      * Better parameter bounds preventing optimization failures
      * Enhanced stability for difficult-to-fit datasets
    * **Technical improvements**: All fitting functions benefit from:
      * Automatic fallback to manual estimation if advanced algorithms fail
      * Consistent error handling and parameter validation
      * Optimized performance with reduced computation time
      * Full integration with DataLab's signal processing workflow
  * Zero padding:
    * Enhanced feature to support both prepending and appending zeros.
    * Partial implementation of [Issue #216](https://github.com/DataLab-Platform/DataLab/issues/216).
    * Default strategy is now "Next power of 2".
  * Ideal frequency domain filter ("Brick wall filter"):
    * It is implemented in `sigima.proc.signal.frequency_filter`.
    * This closes [Issue #215](https://github.com/DataLab-Platform/DataLab/issues/215) - Filters / Brickwall filter (Signal).
  * Bandwidth at -3dB:
    * Enhanced to support passband bandwidth in addition to baseband bandwidth.
  * **Pulse features extraction**: New comprehensive pulse analysis feature for step and square signals:
    * **Automated pulse shape recognition**: Heuristically identifies signal type (step, square, or other) for optimal analysis.
    * **Signal polarity detection**: Automatically determines positive/negative pulse polarity using robust baseline comparison.
    * **Comprehensive parameter extraction**: Measures amplitude, rise time, fall time, full width at half maximum (FWHM), timing parameters (x10, x50, x90, x100), and foot duration.
    * **Baseline characterization**: Extracts start and end baseline ranges for accurate feature computation.
    * **Intelligent baseline estimation**: Uses statistical methods to determine baseline levels from user-defined ranges.
    * **Robust feature computation**: Implements advanced algorithms for timing parameter extraction with noise tolerance.
    * **Visual feedback**: Results displayed in tabular format with pulse feature visualization on the signal plot.
    * This feature is powered by the underlying `sigima.proc.signal.extract_pulse_features` function and supports both step and square pulse analysis.

* New "Extent" group box in image properties:
  * Added computed parameters for image extent: `Xmin`, `Xmax`, `Ymin`, and `Ymax`.
  * These parameters are automatically calculated based on the image origin, pixel spacing, and dimensions.
  * They provide the physical coordinate boundaries of the image for enhanced spatial analysis.
  * The parameters are displayed in a dedicated "Extent" group in the image properties panel.

* New I/O features:
  * Improved delimiter handling in CSV reading (signals) to better support files with variable whitespace separators (now using `\s+` instead of single space character).
  * FT-Lab signals and images:
    * Added support for CEA's FT-Lab signal and image binary formats.
    * This closes [Issue #211](https://github.com/DataLab-Platform/DataLab/issues/211) - Add support for .sig and .ima file types
  * Coordinated text image files (similar to the Matris image format):
    * Supports both real and complex-valued image data with optional error images.
    * Automatically handles NaN values in the data.
    * Reads metadata including units (X, Y, Z) and labels from file headers.
    * This feature is provided by the underlying Sigima library.

* Application settings:
  * Added new "Lock image aspect ratio to 1:1" option in the "Image" section of the "Settings" dialog box:
    * If enabled, the aspect ratio of images is locked to 1:1 (square pixels) when resizing images
    * The default and recommended setting is to disable this option, so that the aspect ratio of images is locked to a value corresponding to the physical pixel size of the image
    * This closes [Issue #244](https://github.com/DataLab-Platform/DataLab/issues/244) - Add option to lock image aspect ratio to 1:1 (disabled by default, so that physical pixel size is used)
  * Added new "Show console on error" option in the "Console" section of the "Settings" dialog box:
    * If enabled, the console will be shown when an error occurs in the application, allowing to see the error traceback (this was previously the default behavior)
    * Now, the console is not shown by default on error, but it does not prevent the console from containing the error traceback as it works in the background.
  * Added configurable autoscale margins for signal and image plots in the "Visualization" section of the "Settings" dialog box:
    * New "Autoscale margin" setting allows customizing the autoscale margin percentage
    * Signal and image plots have separate margin settings for fine-tuned control
    * Margins can be set between 0% and 50% to control the amount of whitespace around data when autoscaling plots
    * Changes are applied immediately to all existing plot widgets without requiring application restart
    * This feature enhances plot visualization control and replaces the previously hardcoded 0.2% margin values

* Dialog sizing improvements:
  * Enhanced initial sizing logic for processing dialogs (profile extraction, offset subtraction, etc.):
    * Dialogs now intelligently resize based on parent window dimensions with proper aspect ratio handling
    * The sizing ratio represents the percentage of parent dimensions, ensuring dialogs are never larger than their parent window
    * Improved user experience with consistently appropriately-sized dialogs across different screen resolutions and window sizes

* Signal cursors:
  * Fixed [Issue #231](https://github.com/DataLab-Platform/DataLab/issues/231) - Add horizontal cursors to the Signal View (requires PlotPy v2.8)
  * Fixed [Issue #232](https://github.com/DataLab-Platform/DataLab/issues/232) - Add Δx measurement on curve stats tool (vertical cursors)

* Computation functions refactoring:
  * Added new `@computation_function` decorator to mark and register computation functions
  * Renamed computation functions to remove redundant "compute_" prefix (e.g., `compute_fft` ➝ `fft`)
  * Added infrastructure for discovering computation functions via introspection
  * Improved function naming in coordinate transformations (e.g., `to_polar` ➝ `to_polar`)
  * Added type hints and improved documentation across computation modules
  * These changes improve code organization and maintainability while making the API more intuitive
  * Internal changes only - **no backward compatibility is maintained** for plugin developers

* Generic processing types refactoring:
  * Refactored the naming convention of core `BaseProcessor` methods implementing generic data processing patterns (e.g. one input ➝ one output, n inputs ➝ one output, etc.)
  * New method names are now based on their input/output signature for better clarity and extensibility:
    * `compute_1_to_1`: transforms each selected object independently
    * `compute_1_to_0`: performs a measurement or analysis without generating a new object
    * `compute_1_to_n`: generates multiple outputs from a single input (e.g. ROI extraction)
    * `compute_n_to_1`: combines multiple inputs into a single output (e.g. sum, mean); supports pairwise mode
    * `compute_2_to_1`: applies a binary operation between an object and a second operand (object or constant); supports pairwise mode
  * These methods are internal and used by advanced developers (e.g. plugin authors); hence, **no backward compatibility is maintained**
  * This closes [Issue #180](https://github.com/DataLab-Platform/DataLab/issues/180) - Rationalize `BaseProcessor` method names for core processing types

* **Enhanced HDF5 file support**: Improved support for HDF5 files with custom file extensions
  * Expanded file dialog filters: File dialogs now include "All files (*)" option alongside standard HDF5 extensions (`.h5`, `.hdf5`, `.hdf`, `.he5`), allowing users to select HDF5 files regardless of their extension
  * Intelligent HDF5 file detection: Added smart file detection that can identify HDF5 files by content, not just extension
    * Extension-based detection (fast): Used for file dialog filtering and standard workflows
    * Content-based detection (thorough): Used for drag-and-drop operations to automatically detect HDF5 files with custom extensions
  * Improved drag-and-drop support: Users can now drag and drop HDF5 files with any extension (e.g., `.data`, `.experiment`, `.custom`) and DataLab will automatically recognize them as HDF5 files if they contain valid HDF5 data
  * Backward compatibility maintained: All existing functionality for standard HDF5 extensions continues to work exactly as before

* HDF5 Browser usability improvements:
  * Changed default tree view state from fully expanded to collapsed for better navigation of large HDF5 files
  * When opening an HDF5 file, only the root node is expanded, keeping all groups and datasets collapsed by default
  * Users can still expand specific nodes of interest, reducing visual clutter and improving performance
  * This behavior matches the "Restore" action which collapses all nodes except the root level

🛠️ Bug fixes:

* **File import ordering**: Fixed inconsistent file ordering across platforms when importing images/signals
  * Files are now explicitly sorted alphabetically in all import methods (dialog, drag-and-drop, directory)
  * Previously, the method used for opening files directly or opening whole directories returned files in filesystem-dependent order (sorted on Windows, arbitrary on Linux/WSL)
  * All import methods now use consistent lexicographic sorting for predictable, platform-independent behavior

* **Plot cleanup robustness**: Fixed ValueError when deleting plot items during result removal
  * The `cleanup_dataview()` method now deletes items one by one with proper error handling
  * Each item is checked for existence before deletion, and ValueError exceptions are caught if the item was already removed
  * **Critical fix**: Result label items are now properly excluded from cleanup to prevent accidental removal
  * Previously, cleanup was removing result labels from the plot, causing all analysis results to disappear when running a second analysis
  * This prevents race conditions where `item.detach()` removes an item from the internal list before the explicit removal call

* Removed "Use reference image LUT range" setting:
  * This feature has been removed to avoid confusion about whether LUT changes were temporary or persistent
  * The feature made users think changes were view-only, but they were actually persistent modifications to image properties
  * **Migration strategy**: Use the new multi-selection property editing instead:
    1. Select multiple images in the object list (Ctrl+click or Shift+click)
    2. Open the Properties panel
    3. Modify the LUT range (Z scale bounds) or other visualization properties
    4. Click Apply to update all selected images simultaneously
  * This approach is more explicit, more powerful (you choose which images to affect), and clearer about persistence
  * The multi-selection feature allows synchronizing not just LUT ranges, but any image property across multiple objects
* Fixed group numbering issue: When creating a new group via "Edit > New Group", the group short ID is now correctly numbered sequentially (e.g., g002) instead of being incorrectly numbered as g000
* Plot interface improvements:
  * Disabled PlotPy's generic "Axes" tab in parameter dialogs to prevent interference with DataLab's own axis scale management.
  * This provides a cleaner, more focused parameter dialog interface while maintaining DataLab's specialized axis handling capabilities.
* Fixed [Issue #233](https://github.com/DataLab-Platform/DataLab/issues/233) - Hard crash when trying to activate the curve stats tool on a zero signal
* Fixed [Issue #184](https://github.com/DataLab-Platform/DataLab/issues/184) - Curve marker style unexpectedly changes to "Square" after validating "Parameters…" dialog
* Fixed [Issue #117](https://github.com/DataLab-Platform/DataLab/issues/117) - DataLab's signal moving median crashes on Linux with `mode='mirror'`: `free(): invalid next size (normal)` (this is a bug in SciPy v1.15.0 to v1.15.2, which was fixed in SciPy v1.15.3)
* Fixed [Issue #186](https://github.com/DataLab-Platform/DataLab/issues/186) - Image text files with a comma as decimal separator cannot be opened directly (only the Import Wizard does support this)
* Fixed [Issue #238](https://github.com/DataLab-Platform/DataLab/issues/238) - Image text files with a trailing delimiter leads to data with a superfluous column when opened directly (does not happen in the Import Wizard)
* Fixed [Issue #239](https://github.com/DataLab-Platform/DataLab/issues/239) - Text Import Wizard does not preserve user-defined titles and units
* Fixed [Issue #240](https://github.com/DataLab-Platform/DataLab/issues/240) - Text Import Wizard does not preserve user-defined data type (e.g. `int16`, `float32`, etc.)
* Fixed [Issue #241](https://github.com/DataLab-Platform/DataLab/issues/241) - Erroneous ifft1d x-axis computation when shift=False
* Improved initial frequency estimate for sinusoidal curve fitting.
* Fixed curve fitting parameter display: added default format string (`%g`) to all `FitParam` instances to ensure proper number formatting in fit dialog parameter fields, preventing display issues with very large or very small parameter values.
* Fixed [Issue #246](https://github.com/DataLab-Platform/DataLab/issues/246) - Incorrect shape unpacking in edit_new_image_parameters
* Fixed [Issue #247](https://github.com/DataLab-Platform/DataLab/issues/247) - Incorrect unpacking of image shape in get_newparam_from_current

🔒 Security fixes:

* **Dependency vulnerability fix**: Fixed CVE-2023-4863 vulnerability in opencv-python-headless
  * Updated minimum requirement from 4.5.4.60 to 4.8.1.78
  * Addresses libwebp binaries vulnerability in bundled OpenCV wheels
  * See [DataLab security advisory](https://github.com/DataLab-Platform/DataLab/security/dependabot/1) for details

ℹ️ Other changes:

* Bumped minimum `plotpy` requirement to V2.8
* Bumped minimum `guidata` requirement to V3.13
* Using new `guidata` translation utility based on `babel`

## DataLab Version 0.20.0 ##

💥 New features and enhancements:

* ANDOR SIF Images:
  * Added support for background images in ANDOR SIF files
  * This closes [Issue #178](https://github.com/DataLab-Platform/DataLab/issues/178) - Add support for ANDOR SIF files with background image
* Array editor (results, signal and image data, ...):
  * New "Copy all" button in the array editor dialog box, to copy all the data in the clipboard, including row and column headers
  * New "Export" button in the array editor dialog box, to export the data in a CSV file, including row and column headers
  * New "Paste" button in the array editor dialog box, to paste the data from the clipboard into the array editor (this feature is not available for read-only data, such as analysis results)
  * The features above require guidata v3.9.0 or later
  * This closes [Issue #174](https://github.com/DataLab-Platform/DataLab/issues/174), [Issue #175](https://github.com/DataLab-Platform/DataLab/issues/175) and [Issue #176](https://github.com/DataLab-Platform/DataLab/issues/176)
* Fourier analysis features ("Processing" menu):
  * New "Zero padding" feature
  * Implementation for signals:
    * Choose a zero padding strategy (Next power of 2, Double the length, Triple the length, Custom length)
    * Or manually set the zero padding length (if "Custom length" is selected)
  * Implementation for images:
    * Choose a zero padding strategy (Next power of 2, Next multiple of 64, Custom length)
    * Or manually set the zero padding row and column lengths (if "Custom length" is selected)
    * Set the position of the zero padding (bottom-right, centered)
  * This closes [Issue #170](https://github.com/DataLab-Platform/DataLab/issues/170) - Fourier analysis: add zero padding feature for signals and images
* Region of Interest (ROI) editor:
  * This concerns the "Edit Regions of Interest" feature for both signals and images
  * New behavior:
    * Signals: the range ROI selection tool is now active by default, and the user can select right away the range of the signal to be used as a ROI
    * Images: the rectangular ROI selection tool is now active by default, and the user can select right away the rectangular ROI to be used as a ROI
    * This closes [Issue #154](https://github.com/DataLab-Platform/DataLab/issues/154) - ROI editor: activate ROI selection tool by default, so that the user can select right away the area to be used as a ROI
  * Added the "Select tool" to editor's toolbar, to allow the user to switch between the "Select" and "Draw" tools easily without having to use the plot toolbar on the top of the window
* Signal processing features ("Processing" menu):
  * New "X-Y mode" feature: this feature simulates the behavior of the X-Y mode of an oscilloscope, i.e. it allows to plot one signal as a function of another signal (e.g. X as a function of Y)
  * New abscissa and ordinate find features:
    * "First abscissa at y=..." feature: this feature allows to find the first abscissa value of a signal at a given y value (e.g. the abscissa value of a signal at y=0)
    * "Ordinate at x=..." feature: this feature allows to find the ordinate value of a signal at a given x value (e.g. the ordinate value of a signal at x=0)
    * Each feature has its own dialog box, which allows to set the y or x value to be used for the search with a slider or a text box
    * This closes [Issue #125](https://github.com/DataLab-Platform/DataLab/issues/125) and [Issue #126](https://github.com/DataLab-Platform/DataLab/issues/126)
  * New full width at given y feature:
    * The "Full width at y=..." feature allows to find the full width of a signal at a given y value (e.g. the full width of a signal at y=0)
    * A specific dialog box allows to set the y value to be used for the search with a slider or a text box
    * This closes [Issue #127](https://github.com/DataLab-Platform/DataLab/issues/127)
* Public API (local or remote):
  * Add `group_id` and `set_current` arguments to `add_signal`, `add_image` and `add_object` methods:
    * This concerns the `LocalProxy`, `AbstractDLControl`, `RemoteClient`, `RemoteServer` and `DLMainWindow` classes
    * `group_id` argument allows to specify the group ID where the signal or image should be added (if not specified, the signal or image is added to the current group)
    * `set_current` argument allows to specify if the signal or image should be set as current after being added (default is `True`)
    * This closes [Issue #151](https://github.com/DataLab-Platform/DataLab/issues/151) - Public API: add a keyword `group_id` to `add_signal` and `add_image`

## DataLab Version 0.19.2 ##

🛠️ Bug fixes:

* Fixed [Issue #172](https://github.com/DataLab-Platform/DataLab/issues/172) - Image profiles: when moving/resizing image, profile plots are not refreshed (fixed in PlotPy v2.7.4)
* Fixed [Issue #173](https://github.com/DataLab-Platform/DataLab/issues/173) - Phase spectrum: add unit (degree) and function reference (`numpy.angle`) to the documentation
* Fixed [Issue #177](https://github.com/DataLab-Platform/DataLab/issues/177) - "Open from directory" feature: unexpected group name (a group named "." is created instead of the root folder name)
* Fixed [Issue #169](https://github.com/DataLab-Platform/DataLab/issues/169) - Signal / Fourier analysis: magnitude spectrum feature does not work as expected with logarithmic scale enabled
* Fixed [Issue #168](https://github.com/DataLab-Platform/DataLab/issues/168) - Average profile visualization: empty profile is displayed when the target rectangular area is outside the image area (this has been fixed upstream, in PlotPy v2.7.4, and so requires the latest version of PlotPy)

## DataLab Version 0.19.1 ##

🛠️ Bug fixes:

* Pairwise operation mode:
  * Fixed an unexpected behavior when using the pairwise operation mode with functions that take a single second operand (e.g. for images: difference, division, arithmetic operations, and flatfield correction)
  * If only one set of operands was selected in a single group, a warning message was displayed "In pairwise mode, you need to select objects in at least two groups.", which is correct for functions that are symmetric (e.g. addition, multiplication, etc.), but not for functions that are not symmetric (e.g. difference, division, etc.).
  * This is now fixed: the warning message is only displayed for functions that are symmetric (e.g. addition, multiplication, etc.).
  * This closes [Issue #157](https://github.com/DataLab-Platform/DataLab/issues/157) - Pairwise operation mode: unexpected behavior with functions that take a single second operand
* Fixed [Issue #152](https://github.com/DataLab-Platform/DataLab/issues/152) - Ignore `nan` values for image normalization, flatfield correction, offset correction, and centroid computation
* Fixed [Issue #153](https://github.com/DataLab-Platform/DataLab/issues/153) - Ignore `nan` values for signal normalization and statistics computations (both analysis result and interactive tool)
* Fixed [Issue #158](https://github.com/DataLab-Platform/DataLab/issues/158) - When editing ROI of a list of images, the first image of the selection is shown (instead of the last as in the image panel)
* Fixed [Issue #159](https://github.com/DataLab-Platform/DataLab/issues/159) - When selecting multiple images just after opening an HDF5 file, the "View in a new window" feature does not work (`KeyError` exception)
* Fixed [Issue #160](https://github.com/DataLab-Platform/DataLab/issues/160) - When selecting multiple images and clearing ROI in ROI editor, only the first image is affected
* Fixed [Issue #161](https://github.com/DataLab-Platform/DataLab/issues/161) - Refresh image items only if necessary (when editing ROI, pasting/deleting metadata)
* Fixed [Issue #162](https://github.com/DataLab-Platform/DataLab/issues/162) - View in a new window: when displaying multiple images, the item list panel should be visible
* Fixed [Issue #163](https://github.com/DataLab-Platform/DataLab/issues/163) - Open from directory: expected one group per folder when loading multiple files
* Fixed [Issue #164](https://github.com/DataLab-Platform/DataLab/issues/164) - Open from directory: unsupported files should be ignored when loading files recursively, to avoid warning popup dialog boxes
* Fixed [Issue #165](https://github.com/DataLab-Platform/DataLab/issues/165) - When opening a file, the default signal/image title must be set to the file name, instead of the relative path to the file name

## DataLab Version 0.19.0 ##

💥 New features and enhancements:

* Image operation features ("Operations" menu):
  * Renamed "Rotation" submenu to "Flip or rotation"
  * New "Flip diagonally" feature
* Signal processing features ("Processing" menu):
  * New "Convert to Cartesian coordinates" feature
  * New "Convert to polar coordinates" feature
* Signal analysis features ("Analysis" menu):
  * Renamed "X values at min/max" to "Abscissa of the minimum and maximum"
  * New "Abscissa at y=..." feature
* New "Open from directory" feature:
  * This feature allows to open multiple files from a directory at once, recursively (only the files with the supported extensions by the current panel are opened)
  * Add "Open from directory" action to the "File" menu for both Signal and Image panels
  * Add support for folders when dropping files in the Signal and Image panels
* Add `1/x` operation to the "Operations" menu for both Signal and Image panels:
  * This feature relies on the `numpy.reciprocal` function, and handles the case where the denominator is zero by catching warnings and replacing the `np.inf` values with `np.nan` values
  * Add `compute_inverse` method for image and signal processors
  * This closes [Issue #143](https://github.com/DataLab-Platform/DataLab/issues/143) - New feature: `1/x` for signals and images
* Public API (local or remote):
  * Add `add_group` method with `title` and `select` arguments to create a new group in a data panel (e.g. Signal or Image panel) and eventually select it after creation:
    * Method was added to the following classes: `AbstractDLControl`, `BaseDataPanel` and `RemoteClient`
    * This closes the following issues:
      * [Issue #131](https://github.com/DataLab-Platform/DataLab/issues/131) - `BaseDataPanel.add_group`: add `select` argument
      * [Issue #47](https://github.com/DataLab-Platform/DataLab/issues/47) - Remote proxy / Public API: add `add_group` method
  * `AbstractDLControl.get_object_uuids`: add an optional `group` argument (group ID, title or number) to eventually filter the objects by group (this closes [Issue #130](https://github.com/DataLab-Platform/DataLab/issues/130))
* When opening an HDF5 file, the confirmation dialog box asking if current workspace should be cleared has a new possible answer "Ignore":
  * Choosing "Ignore" will prevent the confirmation dialog box from being displayed again, and will choose the current setting (i.e. clear or not the workspace) for all subsequent file openings
  * Added a new "Clear workspace before loading HDF5 file" option in the "Settings" dialog box, to allow the user to change the current setting (i.e. clear or not the workspace) for all subsequent file openings
  * Added a new "Ask before clearing workspace" option in the "Settings" dialog box, to allow the user to disable or re-enable the confirmation dialog box asking if current workspace should be cleared when opening an HDF5 file
  * This closes [Issue #146](https://github.com/DataLab-Platform/DataLab/issues/146) - Ask before clearing workspace when opening HDF5 file: add "Ignore" option to prevent dialog from being displayed again
* Object and group title renaming:
  * Removed "Rename group" feature from the "Edit" menu and context menu
  * Added "Rename object" feature to the "Edit" menu and context menu, with F2 shortcut, to rename the title of the selected object or group
  * This closes [Issue #148](https://github.com/DataLab-Platform/DataLab/issues/148) - Rename signal/image/group title by pressing F2
* Region of Interest editor:
  * Regrouped the graphical actions (new rectangular ROI, new circular ROI, new polygonal ROI) in a single menu "Graphical ROI"
  * Added new "Coordinate-based ROI" menu to create a ROI using manual input of the coordinates:
    * For signals, the ROI is defined by the start and end coordinates
    * For images:
      * The rectangular ROI is defined by the top-left and bottom-right coordinates
      * The circular ROI is defined by the center and radius coordinates
      * The polygonal ROI is not supported yet
    * This closes [Issue #145](https://github.com/DataLab-Platform/DataLab/issues/145) - ROI editor: add manual input of the coordinates

🛠️ Bug fixes:

* Fixed [Issue #141](https://github.com/DataLab-Platform/DataLab/issues/141) - Image analysis: mask `nan` values when computing statistics, for example
* Fixed [Issue #144](https://github.com/DataLab-Platform/DataLab/issues/144) - Average profile extraction: `ValueError` when selection rectangle is larger than the image

## DataLab Version 0.18.2 ##

ℹ️ General information:

* Python 3.13 is now supported, since the availability of the scikit-image V0.25 (see [Issue #104](https://github.com/DataLab-Platform/DataLab/issues/104) - Python 3.13: `KeyError: 'area_bbox'`)

💥 Enhancements:

* Added new "Keep results after computation" option in "Processing" section:
  * Before this change, when applying a processing feature (e.g. a filter, a threshold, etc.) on a signal or an image, the analysis results were removed from the object
  * This new option allows to keep the analysis results after applying a processing feature on a signal or an image. Even if the analysis results are not updated, they might be relevant in some use cases (e.g. when using the 2D peak detection feature on an image, and then applying a filter on the image, or summing two images, etc.)

🛠️ Bug fixes:

* Fixed [Issue #138](https://github.com/DataLab-Platform/DataLab/issues/138) - Image colormaps were no longer stored in metadata (and serialized in HDF5 files) since PlotPy v2.6.3 (this commit, specifically: [PlotPyStack/PlotPy@a37af8a](https://github.com/PlotPyStack/PlotPy/commit/a37af8ae8392e5e3655e5c34b67a7cd1544ea845))
* Fixed [Issue #137](https://github.com/DataLab-Platform/DataLab/issues/137) - Arithmetic operations and signal interpolation: dialog box with parameters is not displayed
* Fixed [Issue #136](https://github.com/DataLab-Platform/DataLab/issues/136) - When processing a signal or an image, the analysis result is kept from original object
  * Before this fix, when processing a signal or an image (e.g. when applying a filter, a threshold, etc.), the analysis result was kept from the original object, and was not updated with the new data. Thus the analysis result was not meaningful anymore, and was misleading the user.
  * This is now fixed: the analysis result is now removed when processing a signal or an image. However it is not recalculated automatically, because there is no way to know which analysis result should be recalculated (e.g. if the user has applied a filter, should the FWHM be recalculated?) - besides, the current implementation of the analysis features does not allow to recalculate the analysis results automatically when the data is modified. The user has to recalculate the analysis results manually if needed.
* Fixed [Issue #132](https://github.com/DataLab-Platform/DataLab/issues/132) - Plot analysis results: "One curve per result title" mode ignores ROIs
  * Before this fix, the "One curve per result title" mode was ignoring ROIs, and was plotting the selected result for all objects (signals or images) without taking into account the ROI defined on the objects
  * This is now fixed: the "One curve per result title" mode now takes into account the ROI defined on the objects, and plots the selected result for each object (signal or image) and for each ROI defined on the object
* Fixed [Issue #128](https://github.com/DataLab-Platform/DataLab/issues/128) - Support long object titles in Signal and Image panels
* Fixed [Issue #133](https://github.com/DataLab-Platform/DataLab/issues/133) - Remove specific analysis results from metadata clipboard during copy operation
* Fixed [Issue #135](https://github.com/DataLab-Platform/DataLab/issues/135) - Allow to edit ROI on multiple signals or images at once
  * Before this fix, the ROI editor was disabled when multiple signals or images were selected
  * This is now fixed: the ROI editor is now enabled when multiple signals or images are selected, and the ROI is applied to all selected signals or images (only the ROI of the first selected signal or image is taken into account)
  * This new behavior is consistent with the ROI extraction feature, which allows to extract the ROI on multiple signals or images at once, based on the ROI defined on the first selected signal or image
* Image ROI features:
  * Fixed [Issue #120](https://github.com/DataLab-Platform/DataLab/issues/120) - ROI extraction on multiple images: defined ROI should not be saved in the first selected object. The design choice is to save the defined ROI neither in the first nor in any of the selected objects: the ROI is only used for the extraction, and is not saved in any object
  * Fixed [Issue #121](https://github.com/DataLab-Platform/DataLab/issues/121) - `AttributeError` when extracting multiple ROIs on a single image, if more than one image is selected
  * Fixed [Issue #122](https://github.com/DataLab-Platform/DataLab/issues/122) - Image masks are not refreshed when removing metadata except for the active image
  * Fixed [Issue #123](https://github.com/DataLab-Platform/DataLab/issues/123) - Image masks are not refreshed when pasting metadata on multiple images, except for the last image
* Text and CSV files:
  * Enhance text file reading by detecting data headers (using a list of typical headers from scientific instruments) and by allowing to skip the header when reading the file
  * Ignore encoding errors when reading files in both open feature and import wizard, hence allowing to read files with special characters without raising an exception
  * Fixed [Issue #124](https://github.com/DataLab-Platform/DataLab/issues/124) - Text files: support locale decimal separator (different than `.`)
* Signal analysis features: fixed duplicate results when no ROI is defined
* Fixed [Issue #113](https://github.com/DataLab-Platform/DataLab/issues/113) - Call to `RemoteClient.open_h5_files` (and `import_h5_file`) fails without passing the optional arguments
* Fixed [Issue #116](https://github.com/DataLab-Platform/DataLab/issues/116) - `KeyError` exception when trying to remove a group after opening an HDF5 file

## DataLab Version 0.18.1 ##

💥 Enhancements:

* FWHM computation now raises an exception when less than two points are found with zero-crossing method
* Improved result validation for array-like results by checking the data type of the result

🛠️ Bug fixes:

* Fixed [Issue #106](https://github.com/DataLab-Platform/DataLab/issues/106) - Analysis: coordinate shifted results on images with ROIs and shifted origin
* Fixed [Issue #107](https://github.com/DataLab-Platform/DataLab/issues/107) - Wrong indices when extracting a profile from an image with a ROI
* Fixed [Issue #111](https://github.com/DataLab-Platform/DataLab/issues/111) - Proxy `add_object` method does not support signal/image metadata (e.g. ROI)
* Test data plugin / "Create 2D noisy gauss image": fixed amplitude calculation in `datalab.tests.data.create_2d_random` for non-integer data types

📚 Documentation:

* Fixed path separators in plugin directory documentation
* Corrected left and right area descriptions in workspace documentation
* Updated Google style link in contributing guidelines
* Fixed various French translations in the documentation

## DataLab Version 0.18.0 ##

ℹ️ General information:

* PlotPy v2.7 is required for this release.
* Dropped support for Python 3.8.
* Python 3.13 is not supported yet, due to the fact that some dependencies are not compatible with this version.

💥 New features and enhancements:

* New operation mode feature:
  * Added "Operation mode" feature to the "Processing" tab in the "Settings" dialog box
  * This feature allows to choose between "single" and "pairwise" operation modes for all basic operations (addition, subtraction, multiplication, division, etc.):
    * "Single" mode: single operand mode (default mode: the operation is done on each object independently)
    * "Pairwise" mode: pairwise operand mode (the operation is done on each pair of objects)
  * This applies to both signals and images, and to computations taking *N* inputs
  * Computations taking *N* inputs are the ones where:
    * *N(>=2)* objects in give *N* objects out
    * *N(>=1)* object(s) + 1 object in give N objects out

* New ROI (Region Of Interest) features:
  * New polygonal ROI feature
  * Complete redesign of the ROI editor user interfaces, improving ergonomics and consistency with the rest of the application
  * Major internal refactoring of the ROI system to make it more robust (more tests) and easier to maintain

* Implemented [Issue #102](https://github.com/DataLab-Platform/DataLab/issues/102) - Launch DataLab using `datalab` instead of `datalab`. Note that the `datalab` command is still available for backward compatibility.

* Implemented [Issue #101](https://github.com/DataLab-Platform/DataLab/issues/101) - Configuration: set default image interpolation to anti-aliasing (`5` instead of `0` for nearest). This change is motivated by the fact that a performance improvement was made in PlotPy v2.7 on Windows, which allows to use anti-aliasing interpolation by default without a significant performance impact.

* Implemented [Issue #100](https://github.com/DataLab-Platform/DataLab/issues/100) - Use the same installer and executable on Windows 7 SP1, 8, 10, 11. Before this change, a specific installer was required for Windows 7 SP1, due to the fact that Python 3.9 and later versions are not supported on this platform. A workaround was implemented to make DataLab work on Windows 7 SP1 with Python 3.9.

🛠️ Bug fixes:

* Fixed [Issue #103](https://github.com/DataLab-Platform/DataLab/issues/103) - `proxy.add_annotations_from_items`: circle shape color seems to be ignored.

## DataLab Version 0.17.1 ##

ℹ️ PlotPy v2.6.2 is required for this release.

💥 New features and enhancements:

* Image View:
  * Before this release, when selecting a high number of images (e.g. when selecting a group of images), the application was very slow because all the images were displayed in the image view, even if they were all superimposed on the same image
  * The workaround was to enable the "Show first only" option
  * Now, to improve performance, if multiple images are selected, only the last image of the selection is displayed in the image view if this last image has no transparency and if the other images are completely covered by this last image
* Clarification: action "Show first only" was renamed to "Show first object only", and a new icon was added to the action
* API: added `width` and `height` properties to `ImageObj` class (returns the width and height of the image in physical units)
* Windows launcher "start.pyw": writing a log file "datalab_error.log" when an exception occurs at startup

🛠️ Bug fixes:

* Changing the color theme now correctly updates all DataLab's user interface components without the need to restart the application

ℹ️ Other changes:

* OpenCV is now an optional dependency:
  * This change is motivated by the fact that the OpenCV conda package is not maintained on Windows (at least), which leads to an error when installing DataLab with conda
  * When OpenCV is not installed, only the "OpenCV blob detection" feature won't work, and a warning message will be displayed when trying to use this feature

## DataLab Version 0.17.0 ##

ℹ️ PlotPy v2.6 is required for this release.

💥 New features and enhancements:

* Menu "Computing" was renamed to "Analysis" for both Signal and Image panels, to better reflect the nature of the features in this menu
* Regions Of Interest (ROIs) are now taken into account everywhere in the application where it makes sense, and not only for the old "Computing" menu (now "Analysis") features. This closes [Issue #93](https://github.com/DataLab-Platform/DataLab/issues/93). If a signal or an image has an ROI defined:
  * Operations are done on the ROI only (except if the operation changes the data shape, or the pixel size for images)
  * Processing features are done on the ROI only (if the destination object data type is compatible with the source object data type, which excludes thresholding, for instance)
  * Analysis features are done on the ROI only, like before
* As a consequence of previous point, and for clarity:
  * The "Edit Regions of interest" and "Remove all Regions of interest" features have been moved from the old "Computing" (now "Analysis") menu to the "Edit" menu where all metadata-related features are located
  * The "Edit Regions of interest" action has been added to both Signal and Image View vertical toolbars (in second position, after the "View in a new window" action)
* Following the bug fix on image data type conversion issues with basic operations, a new "Arithmetic operation" feature has been added to the "Operations" menu for both Signal and Image panels. This feature allows to perform linear operations on signals and images, with the following operations:
  * Addition: ``obj3 = (obj1 + obj2) * a + b``
  * Subtraction: ``obj3 = (obj1 - obj2) * a + b``
  * Multiplication: ``obj3 = (obj1 * obj2) * a + b``
  * Division: ``obj3 = (obj1 / obj2) * a + b``
* Improved "View in a new window" and "ROI editor" dialog boxes size management: default size won't be larger than DataLab's main window size
* ROI editor:
  * Added toolbars for both Signal and Image ROI editors, to allow to zoom in and out, and to reset the zoom level easily
  * Rearranged the buttons in the ROI editor dialog box for better ergonomics and consistency with the Annotations editor ("View in a new window" dialog box)
* Application color theme:
  * Added support for color theme (auto, light, dark) in the "Settings" dialog box
  * The color theme is applied without restarting the application

🛠️ Bug fixes:

* Intensity profile / Segment profile extraction:
  * When extracting a profile on an image with a ROI defined, the associated PlotPy feature show a warning message ('UserWarning: Warning: converting a masked element to nan.') but the profile is correctly extracted and displayed, with NaN values where the ROI is not defined.
  * NaN values are now removed from the profile before plotting it
* Simple processing features with a one-to-on mapping with a Python function (e.g. `numpy.absolute`, `numpy.log10`, etc.) and without parameters: fix result object title which was systematically ending with "|" (the character that usually precedes the list of parameters)
* Butterworth filter: fix cutoff frequency ratio default value and valid range
* Fix actions refresh issue in Image View vertical toolbar:
  * When starting DataLab with the Signal Panel active, switching to the Image View was showing "View in a new window" or "Edit Regions of interest" actions enabled in the vertical toolbar, even if no image was displayed in the Image View
  * The Image View vertical toolbar is now correctly updated at startup
* View in a new window: cross section tools (intensity profiles) stayed disabled unless the user selected an image through the item list - this is now fixed
* Image View: "Show contrast panel" toolbar button was not enabled at startup, and was only enabled when at least one image was displayed in the Image View - it is now always enabled, as expected
* Image data type conversion:
  * Previously, the data type conversion feature was common to signal and image processing features, i.e. a simple conversion of the data type using NumPy's `astype` method
  * This was not sufficient for image processing features, in particular for integer images, because even if the result was correct from a numerical point of view, underflow or overflow could be legitimately seen as a bug from a mathematical point of view
  * The image data type conversion feature now relies on the internal `clip_astype` function, which clips the data to the valid range of the target data type before converting it (in the case of integer images)
* Image ROI extraction issues:
  * Multiple regressions were introduced in version 0.16.0:
    * Single circular ROI extraction was not working as expected (a rectangular ROI was extracted, with unexpected coordinates)
    * Multiple circular ROI extraction lead to a rectangular ROI extraction
    * Multiple ROI extraction was no longer cropping the image to the overall bounding box of the ROIs
  * These issues are now fixed, and unit tests have been added to prevent regressions:
    * An independent test algorithm has been implemented to check the correctness of the ROI extraction in all cases mentioned above
    * Tests cover both single and multiple ROI extraction, with circular and rectangular ROIs
* Overflow and underflow issues in some operations on integer images:
  * When processing integer images, some features were causing overflow or underflow issues, leading to unexpected results (correct results from a numerical point of view, but not from a mathematical point of view)
  * This issue only concerned basic operations (addition, subtraction, multiplication, division, and constant operations) - all the other features were already working as expected
  * This is now fixed as result output are now floating point images
  * Unit tests have been added to prevent regressions for all these operations

## DataLab Version 0.16.4 ##

This is a minor maintenance release.

🛠️ Bug fixes:

* Requires PlotPy v2.4.1 or later to fix the following issues related to the contrast adjustment feature:
  * A regression was introduced in an earlier version of PlotPy: levels histogram was no longer removed from contrast adjustment panel when the associated image was removed from the plot
  * This is now fixed: when an image is removed, the histogram is removed as well and the contrast panel is refreshed (which was not the case even before the regression)
* Ignore `AssertionError` in *config_unit_test.py* when executing test suite on WSL

📚 Documentation:

* Fix class reference in `Wrap11Func` documentation

## DataLab Version 0.16.3 ##

🛠️ Bug fixes:

* Fixed [Issue #84](https://github.com/DataLab-Platform/DataLab/issues/84) - Build issues with V0.16.1: `signal` name conflict, ...
  * This issue was intended to be fixed in version 0.16.2, but the fix was not complete
  * Thanks to [@rolandmas](https://github.com/rolandmas) for reporting the issue and for the help in investigating the problem and testing the fix
* Fixed [Issue #85](https://github.com/DataLab-Platform/DataLab/issues/85) - Test data paths may be added multiple times to `datalab.utils.tests.TST_PATH`
  * This issue is related to [Issue #84](https://github.com/DataLab-Platform/DataLab/issues/84)
  * Adding the test data paths multiple times to `datalab.utils.tests.TST_PATH` was causing the test data to be loaded multiple times, which lead to some tests failing (a simple workaround was added to V0.16.2: this issue is now fixed)
  * Thanks again to [@rolandmas](https://github.com/rolandmas) for reporting the issue in the context of the Debian packaging
* Fixed [Issue #86](https://github.com/DataLab-Platform/DataLab/issues/86) - Average of N integer images overflows data type
* Fixed [Issue #87](https://github.com/DataLab-Platform/DataLab/issues/87) - Image average profile extraction: `AttributeError` when trying to edit profile parameters
* Fixed [Issue #88](https://github.com/DataLab-Platform/DataLab/issues/88) - Image segment profile: point coordinates inversion

## DataLab Version 0.16.2 ##

This release requires PlotPy v2.4.0 or later, which brings the following bug fixes and new features:

* New constrast adjustment features and bug fixes:
  * New layout: the vertical toolbar (which was constrained in a small area on the right side of the panel) is now a horizontal toolbar at the top of the panel, beside the title
  * New "Set range" button: allows the user to set manually the minimum and maximum values of the histogram range
  * Fixed histogram update issues when no image was currently selected (even if the an image was displayed and was selected before)
  * Histogram range was not updated when either the minimum or maximum value was set using the "Minimum value" or "Maximum value" buttons (which have been renamed to "Min." and "Max." in this release)
  * Histogram range was not updated when the "Set full range" button was clicked, or when the LUT range was modified using the "Scales / LUT range" form in "Properties" group box

* Image view context menu: new "Reverse X axis" feature

ℹ️ Minor new features and enhancements:

* Image file types:
  * Added native support for reading .SPE, .GEL, .NDPI and .REC image files
  * Added support for any `imageio`-supported file format through configuration file (entry `imageio_formats` may be customized to complement the default list of supported formats: see [documentation](https://datalab-platform.com/en/features/image/menu_file.html#open-image) for more details)

🛠️ Bug fixes:

* Image Fourier analysis:
  * Fixed logarithmic scale for the magnitude spectrum (computing dB instead of natural logarithm)
  * Fixed PSD computation with logarithmic scale (computing dB instead of natural logarithm)
  * Updated the documentation to explicitly mention that the logarithmic scale is in dB

* Fixed [Issue #82](https://github.com/DataLab-Platform/DataLab/issues/82) - Macros are not renamed in DataLab after exporting them to Python scripts

* `ResultProperties` object can now be added to `SignalObj` or `ImageObj` metadata even outside a Qt event loop (because the label item is no longer created right away)

* Progress bar is now automatically closed as expected when an error occurrs during a long operation (e.g. when opening a file)

* Difference, division...: dialog box for the second operand selection was allowing to select a group (only a signal or an image should be selected)

* When doing an operation which involves an object (signal or image) with higher order number than the current object (e.g. when subtracting an image with an image from a group below the current image), the resulting object's title now correctly refers to the order numbers of the objects involved in the operation (e.g., to continue with the subtraction example mentioned above, the resulting object's title was previously referring to the order number before the insertion of the resulting image)

* Added support for additional test data folder thanks to the `DATALAB_DATA` environment variable (useful for testing purposes, and especially in the context of Debian packaging)

## DataLab Version 0.16.1 ##

Since version 0.16.0, many validation functions have been added to the test suite. The percentage of validated compute functions has increased from 37% to 84% in this release.

NumPy 2.0 support has been added with this release.

ℹ️ Minor new features and enhancements:

* Signal and image moving average and median filters:
  * Added "Mode" parameter to choose the mode of the filter (e.g. "reflect", "constant", "nearest", "mirror", "wrap")
  * The default mode is "reflect" for moving average and "nearest" for moving median
  * This allows to handle edge effects when filtering signals and images

🛠️ Bug fixes:

* Fixed Canny edge detection to return binary image as `uint8` instead of `bool` (for consistency with other image processing features)

* Fixed Image normalization: lower bound was wrongly set for `maximum` method

* Fixed `ValueError` when computing PSD with logarithmic scale

* Fixed Signal derivative algorithm: now using `numpy.gradient` instead of a custom implementation

* Fixed SciPy's `cumtrapz` deprecation: use `cumulative_trapezoid` instead

* Curve selection now shows the individual points of the curve (before, only the curve line width was broadened)

* Windows installer: add support for unstable releases (e.g., 0.16.1.dev0), thus allowing to easily install the latest development version of DataLab on Windows

* Fixed [Issue #81](https://github.com/DataLab-Platform/DataLab/issues/81) - When opening files, show progress dialog only if necessary

* Fixed [Issue #80](https://github.com/DataLab-Platform/DataLab/issues/80) - Plotting results: support for two use cases
  * The features of the "Analysis" menu produce *results* (scalars): blob detection (circle coordinates), 2D peak detection (point coordinates), etc. Depending on the feature, result tables are displayed in the "Results" dialog box, and the results are also stored in the signal or image metadata: each line of the result table is an individual result, and each column is a property of the result - some results may consist only of a single individual result (e.g., image centroid or curve FHWM), while others may consist of multiple individual results (e.g., blob detection, contour detection, etc.).
  * Before this change, the "Plot results" feature only supported plotting the first individual result of a result table, as a function of the index (of the signal or image objects) or any of the columns of the result table. This was not sufficient for some use cases, where the user wanted to plot multiple individual results of a result table.
  * Now, the "Plot results" feature supports two use cases:
    * "One curve per result title": Plotting the first individual result of a result table, as before
    * "One curve per object (or ROI) and per result title": Plotting all individual results of a result table, as a function of the index (of the signal or image objects) or any of the columns of the result table
  * The selection of the use case is done in the "Plot results" dialog box
  * The default use case is "One curve per result title" if the result table has only one line, and "One curve per object (or ROI) and per result title" otherwise

## DataLab Version 0.16.0 ##

💥 New features and enhancements:

* Major user interface overhaul:
  * The menu bar and toolbars have been reorganized to make the application more intuitive and easier to use
  * Operations and processing features have been regrouped in submenus
  * All visualization-related actions are now grouped in the plot view vertical toolbar
  * Clarified the "Annotations" management (new buttons, toolbar action...)

* New validation process for signal and image features:
  * Before this release, DataLab's validation process was exclusively done from the programmer's point of view, by writing unit tests and integration tests, thus ensuring that the code was working as expected (i.e. that no exception was raised and that the behavior was correct)
  * With this release, a new validation process has been introduced, from the user's point of view, by adding new validation functions (marked with the `@pytest.mark.validation` decorator) in the test suite
  * A new "Validation" section in the documentation explains how validation is done and contains a list of all validation functions with the statistics of the validation process (generated from the test suite)
  * The validation process is a work in progress and will be improved in future versions

* "Properties" group box:
  * Added "Scales" tab, to show and set the plot scales:
    * X, Y for signals
    * X, Y, Z (LUT range) for images

* View options:
  * New "Show first only" option in the "View" menu, to show only the first curve (or image) when multiple curves (or images) are displayed in the plot view
  * New (movable) label for FWHM computations, additional to the existing segment annotation

* I/O features:
  * Added support for reading and writing .MAT files (MATLAB format)
  * Create a new group when opening a file containing multiple signals or images (e.g. CSV file with multiple curves)

* Add support for binary images
* Signal ROI extraction: added new dialog box to manually edit the ROI lower and upper bounds after defining the ROI graphically

ℹ️ New **Signal** operations, processing and analysis features:

| Menu        | Submenu      |Features                                                 |
|-------------|--------------|---------------------------------------------------------|
| New | New signal | Exponential, pulse, polynomial, custom (manual input)            |
| Operations  | | Exponential, Square root, Power |
| Operations  | Operations with a constant | +, -, *, / |
| Processing  | Axis Transformation | Reverse X-axis |
| Processing  | Level Adjustment | Offset correction |
| Processing  | Fourier analysis | Power spectrum, Phase spectrum, Magnitude spectrum, Power spectral density |
| Processing  | Frequency filters | Low-pass, High-pass, Band-pass, Band-stop |
| Processing  | | Windowing (Hanning, Hamming, Blackman, Blackman-Harris, Nuttall, Flat-top...) |
| Processing  | Fit | Linear fit, Sinusoidal fit, Exponential fit, CDF fit |
| Analysis   | | FWHM (Zero-crossing method), X value @ min/max, Sampling period/frequency, Dynamic parameters (ENOB, SNR, SINAD, THD, SFDR), -3dB bandwidth, Contrast |

ℹ️ New **Image** operations, processing and analysis features:

| Menu        | Submenu      |Features                                                 |
|-------------|--------------|---------------------------------------------------------|
| Operations  | | Exponential |
| Operations  | Intensity profiles | Profile along a segment |
| Operations  | Operations with a constant | +, -, *, / |
| Processing  | Level Adjustment | Normalization, Clipping, Offset correction |
| Processing  | Fourier analysis | Power spectrum, Phase spectrum, Magnitude spectrum, Power spectral density |
| Processing  | Thresholding | Parametric, ISODATA, Li, Mean, Minimum, Otsu, Triangle, Yen |

🛠️ Bug fixes:

* Fixed a performance issue due to an unnecessary refresh of the plot view when adding a new signal or image
* Fixed [Issue #77](https://github.com/DataLab-Platform/DataLab/issues/77) - Intensity profiles: unable to accept dialog the second time
* Fixed [Issue #75](https://github.com/DataLab-Platform/DataLab/issues/75) - View in a new window: curve anti-aliasing is not enabled by default
* Annotations visibility is now correctly saved and restored:
  * Before this release, when modifying the annotations visibility in the separate plot view, the visibility was not saved and restored when reopening the plot view
  * This has been [fixed upstream](https://github.com/PlotPyStack/PlotPy/commit/03faaa42e5d6d4016ea8c99334c29d46a5963467) in PlotPy (v2.3.3)

## DataLab Version 0.15.1 ##

🛠️ Bug fixes:

* Fixed [Issue #68](https://github.com/DataLab-Platform/DataLab/issues/68) - Slow loading of even simple plots:
  * On macOS, the user experience was degraded when handling even simple plots
  * This was due to the way macOS handles the pop-up windows, e.g. when refreshing the plot view ("Creating plot items" progress bar), hence causing a very annoying flickering effect and a global slowdown of the application
  * This is now fixed by showing the progress bar only after a short delay (1s), that is when it is really needed (i.e. for long operations)
  * Thanks to [@marcel-goldschen-ohm](https://github.com/marcel-goldschen-ohm) for the very thorough feedback and the help in testing the fix
* Fixed [Issue #69](https://github.com/DataLab-Platform/DataLab/issues/69) - Annotations should be read-only in Signal/Image View
  * Regarding the annotations, DataLab's current behavior is the following:
    * Annotations are created only when showing the signal/image in a separate window (double-click on the object, or "View" > "View in a new window")
    * When displaying the objects in either the "Signal View" or the "Image View", the annotations should be read-only (i.e. not movable, nor resizable or deletable)
  * However, some annotations were still deletable in the "Signal View" and the "Image View": this is now fixed
  * Note that the fact that annotations can't be created in the "Signal View" or the "Image View" is a limitation of the current implementation, and may be improved in future versions

## DataLab Version 0.15.0 ##

🎁 New installer for the stand-alone version on Windows:

* The stand-alone version on Windows is now distributed as an MSI installer (instead of an EXE installer)
* This avoids the false positive detection of the stand-alone version as a potential threat by some antivirus software
* The program will install files and shortcuts:
  * For current user, if the user has no administrator privileges
  * For all users, if the user has administrator privileges
  * Installation directory may be customized
* MSI installer allows to integrate DataLab's installation seemlessly in an organization's deployment system

💥 New features and enhancements:

* Added support for large text/CSV files:
  * Files over 1 GB (and with reasonable number of lines) can now be imported as signals or images without crashing the application or even slowing it down
  * The file is read by chunks and, for signals, the data is downsampled to a reasonable number of points for visualization
  * Large files are supported when opening a file (or dragging and dropping a file in the Signal Panel) and when importing a file in the Text Import Wizard
* Auto downsampling feature:
  * Added "Auto downsampling" feature to signal visualization settings (see "Settings" dialog box)
  * This feature allows to automatically downsample the signal data for visualization when the number of points is too high and would lead to a slow rendering
  * The downsampling factor is automatically computed based on the configured maximum number of points to display
  * This feature is enabled by default and may be disabled in the signal visualization settings
* CSV format handling:
  * Improved support for CSV files with a header row (column names)
  * Added support for CSV files with empty columns
* Open/save file error handling:
  * Error messages are now more explicit when opening or saving a file fails
  * Added a link to the folder containing the file in the error message
* Added "Plugins and I/O formats" page to the Installation and Configuration Viewer (see "Help" menu)
* Reset DataLab configuration:
  * In some cases, it may be useful to reset the DataLab configuration file to its default values (e.g. when the configuration file is corrupted)
  * Added new `--reset` command line option to remove the configuration folder
  * Added new "Reset DataLab" Start Menu shortcut to the Windows installer

🛠️ Bug fixes:

* Fixed [Issue #64](https://github.com/DataLab-Platform/DataLab/issues/64) - HDF5 browser does not show datasets with 1x1 size:
  * HDF5 datasets with a size of 1x1 were not shown in the HDF5 browser
  * Even if those datasets should not be considered as signals or images, they are now shown in the HDF5 browser (but not checkable, i.e. not importable as signals or images)

## DataLab Version 0.14.2 ##

⚠️ API changes required for fixing support for multiple signals loading feature:

* Merged `open_object` and `open_objects` methods to `load_from_files` in proxy classes, main window and data panels
* For consistency's sake: merged `save_object` and `save_objects` into `save_to_files`
* To sum up, those changes lead to the following situation:
  * `load_from_files`: load a sequence of objects from multiple files
  * `save_to_files`: save a sequence of objects to multiple files (at the moment, it only supports saving a single object to a single file, but it may be extended in the future to support saving multiple objects to a single file)

🛠️ Bug fixes:

* Fixed [Issue #61](https://github.com/DataLab-Platform/DataLab/issues/61) - Text file import wizard: application crash when importing a multiple curve text file:
  * This issue concerns a use case where the text file contains multiple curves
  * This is now fixed and an automatic test has been added to prevent regressions

## DataLab Version 0.14.1 ##

🎉 New domain name: [datalab-platform.com](https://datalab-platform.com)

💥 New features:

* Added support for colormap inversion in Image View:
  * New "Invert colormap" entry in plot context menu, image parameters, and in the default image view settings
  * This requires `PlotPy` v2.3 or later
* HDF5 Browser:
  * Added "Show array" button at the corner of the "Group" and "Attributes" tabs, to show the array in a separate window (useful for copy/pasting data to other applications, for instance)
  * Attributes: added support for more scalar data types
* Testability and maintainability:
  * DataLab's unit tests are now using [pytest](https://pytest.org). This has required a lot of work for the transition, especially to readapt the tests so that they may be executed in the same process. For instance, a particular attention has been given to sandboxing the tests, so that they do not interfere with each other.
  * Added continuous integration (CI) with GitHub Actions
  * For this release, test coverage is 87%
* Text file import assistant:
  * Drastically improved the performance of the array preview when importing large text files (no more progress bar, and the preview is now displayed almost instantaneously)

🛠️ Bug fixes:

* XML-RPC server was not shut down properly when closing DataLab
* Fixed test-related issues: some edge cases were hidden by the old test suite, and have been revealed by the transition to `pytest`. This has led to some bug fixes and improvements in the code.
* On Linux, when running a computation on a signal or an image, and on rare occasions, the computation was stuck as if it was running indefinitely. Even though the graphical user interface was still responsive, the computation was not progressing and the user had to cancel the operation and restart it. This was due to the start method of the separate process used for the computation (default method was "fork" on Linux). This is now fixed by using the "spawn" method instead, which is the recommended method for latest versions of Python on Linux when multithreading is involved.
* Fixed [Issue #60](https://github.com/DataLab-Platform/DataLab/issues/60) - `OSError: Invalid HDF5 file [...]` when trying to open an HDF5 file with an extension other than ".h5"
* Image Region of Interest (ROI) extraction: when modifying the image bounds in the confirmation dialog box, the ROI was not updated accordingly until the operation was run again
* Deprecation issues:
  * Fixed `scipy.ndimage.filters` deprecation warning
  * Fixed `numpy.fromstring` deprecation warning

## DataLab Version 0.14.0 ##

💥 New features:

* New "Histogram" feature in "Analysis" menu:
  * Added histogram computation feature for both signals and images
  * The histogram is computed on the regions of interest (ROI) if any, or on the whole signal/image if no ROI is defined
  * Editable parameters: number of bins, lower and upper bounds
* HDF5 browser:
  * Improved tree view layout (more compact and readable)
  * Multiple files can now be opened at once, using the file selection dialog box
  * Added tabs with information below the graphical preview:
    * Group info: path, textual preview, etc.
    * Attributes info: name, value
  * Added "Show only supported data" check box: when checked, only supported data (signals and images) are shown in the tree view
  * Added "Show values" check box, to show/hide the values in the tree view
* Macro Panel:
  * Macro commands are now numbered, starting from 1, like signals and images
* Remote control API (`RemoteProxy` and `LocalProxy`):
  * `get_object_titles` method now accepts "macro" as panel name and returns the list of macro titles
  * New `run_macro`, `stop_macro` and `import_macro_from_file` methods

🛠️ Bug fixes:

* Stand-alone version - Integration in Windows start menu:
  * Fixed "Uninstall" shortcut (unclickable due to a generic name)
  * Translated "Browse installation directory" and "Uninstall" shortcuts
* Fixed [Issue #55](https://github.com/DataLab-Platform/DataLab/issues/55) - Changing image bounds in Image View has no effect on the associated image object properties
* Fixed [Issue #56](https://github.com/DataLab-Platform/DataLab/issues/56) - "Test data" plugin: `AttributeError: 'NoneType' object has no attribute 'data'` when canceling "Create image with peaks"
* Fixed [Issue #57](https://github.com/DataLab-Platform/DataLab/issues/57) - Circle and ellipse result shapes are not transformed properly
* Curve color and style cycle:
  * Before this release, this cycle was handled by the same mechanism either for the Signal Panel or the HDF5 Browser, which was not the expected behavior
  * Now, the cycle is handled separately: the HDF5 Browser or the Text Import Wizard use always the same color and style for curves, and they don't interfere with the Signal Panel cycle

## DataLab Version 0.12.0 ##

🧹 Clarity-Enhanced Interface Update:

* The tabs used to switch between the data panels (signals and images) and the visualization components ("Curve panel" and "Image panel") have been renamed to "Signal Panel" and "Image Panel" (instead of "Signals" and "Images")
* The visualization components have been renamed to "Signal View" and "Image View" (instead of "Curve panel" and "Image panel")
* The data panel toolbar has been renamed to "Signal Toolbar" and "Image Toolbar" (instead of "Signal Processing Toolbar" and "Image Processing Toolbar")
* Ergonomics improvements: the "Signal Panel" and "Image Panel" are now displayed on the left side of the main window, and the "Signal View" and "Image View" are displayed on the right side of the main window. This reduces the distance between the list of objects (signals and images) and the associated actions (toolbars and menus), and makes the interface more intuitive and easier to use

✨ New tour and demo feature:

* When starting DataLab for the first time, an optional tour is now shown to the user to introduce the main features of the application
* The tour can be started again at any time from the "?" menu
* Also added a new "Demo" feature to the "?" menu

🚀 New Binder environment to test DataLab online without installing anything

📚 Documentation:

* New text tutorials are available:
  * Measuring Laser Beam Size
  * DataLab and Spyder: a perfect match
* "Getting started" section: added more explanations and links to the tutorials
* New "Contributing" section explaining how to contribute to DataLab, whether you are a developer or not
* New "Macros" section explaining how to use the macro commands feature
* Added "Copy" button to code blocks in the documentation

💥 New features:

* New "Text file import assistant" feature:
  * This feature allows to import text files as signals or images
  * The user can define the source (clipboard or texte file)
  * Then, it is possible to define the delimiter, the number of rows to skip, the destination data type, etc.
* Added menu on the "Signal Panel" and "Image Panel" tabs corner to quickly access the most used features (e.g. "Add", "Remove", "Duplicate", etc.)
* Intensity profile extraction feature:
  * Added graphical user interface to extract intensity profiles from images, for both line and averaged profiles
  * Parameters are still directly editable by the user ("Edit profile parameters" button)
  * Parameters are now stored from one profile extraction to another
* Statistics feature:
  * Added `<y>/σ(y)` to the signal "Statistics" result table (in addition to the mean, median, standard deviation, etc.)
  * Added `peak-to-peak` to the signal and image "Statistics" result table
* Curve fitting feature: fit results are now stored in a dictionary in the signal metadata (instead of being stored individually in the signal metadata)
* Window state:
  * The toolbars and dock widgets state (visibility, position, etc.) are now stored in the configuration file and restored at startup (size and position were already stored and restored)
  * This implements part of [Issue #30](https://github.com/DataLab-Platform/DataLab/issues/30) - Save/restore main window layout

🛠️ Bug fixes:

* Fixed [Issue #41](https://github.com/DataLab-Platform/DataLab/issues/41) - Radial profile extraction: unable to enter user-defined center coordinates
* Fixed [Issue #49](https://github.com/DataLab-Platform/DataLab/issues/49) - Error when trying to open a (UTF-8 BOM) text file as an image
* Fixed [Issue #51](https://github.com/DataLab-Platform/DataLab/issues/51) - Unexpected dimensions when adding new ROI on an image with X/Y arbitrary units (not pixels)
* Improved plot item style serialization management:
  * Before this release, the plot item style was stored in the signal/image metadata only when saving the workspace to an HDF5 file. So, when modifying the style of a signal/image from the "Parameters" button (view toolbar), the style was not kept in some cases (e.g. when duplicating the signal/image).
  * Now, the plot item style is stored in the signal/image metadata whenever the style is modified, and is restored when reloading the workspace
* Handled `ComplexWarning` cast warning when adding regions of interest (ROI) to a signal with complex data

## DataLab Version 0.11.0 ##

💥 New features:

* Signals and images may now be reordered in the tree view:
  * Using the new "Move up" and "Move down" actions in the "Edit" menu (or using the corresponding toolbar buttons):
  * This fixes [Issue #22](https://github.com/DataLab-Platform/DataLab/issues/22) - Add "move up/down" actions in "Edit" menu, for signals/images and groups
* Signals and images may also be reordered using drag and drop:
  * Signals and images can be dragged and dropped inside their own panel to change their order
  * Groups can also be dragged and dropped inside their panel
  * The feature also supports multi-selection (using the standard Ctrl and Shift modifiers), so that multiple signals/images/groups can be moved at once, not necessarily with contiguous positions
  * This fixes [Issue #17](https://github.com/DataLab-Platform/DataLab/issues/17) - Add Drag and Drop feature to Signals/Images tree views
* New 1D interpolation features:
  * Added "Interpolation" feature to signal panel's "Processing" menu
  * Methods available: linear, spline, quadratic, cubic, barycentric and PCHIP
  * Thanks to [@marcel-goldschen-ohm](https://github.com/marcel-goldschen-ohm) for the contribution to spline interpolation
  * This fixes [Issue #20](https://github.com/DataLab-Platform/DataLab/issues/20) - Add 1D interpolation features
* New 1D resampling feature:
  * Added "Resampling" feature to signal panel's "Processing" menu
  * Same interpolation methods as for the "Interpolation" feature
  * Possibility to specify the resampling step or the number of points
  * This fixes [Issue #21](https://github.com/DataLab-Platform/DataLab/issues/21) - Add 1D resampling feature
* New 1D convolution feature:
  * Added "Convolution" feature to signal panel's "Operation" menu
  * This fixes [Issue #23](https://github.com/DataLab-Platform/DataLab/issues/23) - Add 1D convolution feature
* New 1D detrending feature:
  * Added "Detrending" feature to signal panel's "Processing" menu
  * Methods available: linear or constant
  * This fixes [Issue #24](https://github.com/DataLab-Platform/DataLab/issues/24) - Add 1D detrending feature
* 2D analysis results:
  * Before this release, 2D analysis results such as contours, blobs, etc. were stored in image metadata dictionary as coordinates (x0, y0, x1, y1, ...) even for circles and ellipses (i.e. the coordinates of the bounding rectangles).
  * For convenience, the circle and ellipse coordinates are now stored in image metadata dictionary as (x0, y0, radius) and (x0, y0, a, b, theta) respectively.
  * These results are also shown as such in the "Results" dialog box (either at the end of the computing process or when clicking on the "Show results" button).
  * This fixes [Issue #32](https://github.com/DataLab-Platform/DataLab/issues/32) - Contour detection: show circle `(x, y, r)` and ellipse `(x, y, a, b, theta)` instead of `(x0, y0, x1, x1, ...)`
* 1D and 2D analysis results:
  * Additionnaly to the previous enhancement, more analysis results are now shown in the "Results" dialog box
  * This concerns both 1D (FHWM...) and 2D analysis results (contours, blobs...):
    * Segment results now also show length (L) and center coordinates (Xc, Yc)
    * Circle and ellipse results now also show area (A)
* Added "Plot results" entry in "Analysis" menu:
  * This feature allows to plot analysis results (1D or 2D)
  * It creates a new signal with X and Y axes corresponding to user-defined parameters (e.g. X = indices and Y = radius for circle results)
* Increased default width of the object selection dialog box:
  * The object selection dialog box is now wider by default, so that the full signal/image/group titles may be more easily readable
* Delete metadata feature:
  * Before this release, the feature was deleting all metadata, including the Regions Of Interest (ROI) metadata, if any.
  * Now a confirmation dialog box is shown to the user before deleting all metadata if the signal/image has ROI metadata: this allows to keep the ROI metadata if needed.
* Image profile extraction feature: added support for masked images (when defining regions of interest, the areas outside the ROIs are masked, and the profile is extracted only on the unmasked areas, or averaged on the unmasked areas in the case of average profile extraction)
* Curve style: added "Reset curve styles" in "View" menu. This feature allows to reset the curve style cycle to its initial state.
* Plugin base classe `PluginBase`:
  * Added `edit_new_signal_parameters` method for showing a dialog box to edit parameters for a new signal
  * Added `edit_new_image_parameters` method for showing a dialog box to edit parameters for a new image (updated the *datalab_testdata.py* plugin accordingly)
* Signal and image computations API (`datalab.computations`):
  * Added wrappers for signal and image 1 -> 1 computations
  * These wrappers aim at simplifying the creation of a basic computation function operating on DataLab's native objects (`SignalObj` and `ImageObj`) from a function operating on NumPy arrays
  * This simplifies DataLab's internals and makes it easier to create new computing features inside plugins
  * See the *datalab_custom_func.py* example plugin for a practical use case
* Added "Radial profile extraction" feature to image panel's "Operation" menu:
  * This feature allows to extract a radially averaged profile from an image
  * The profile is extracted around a user-defined center (x0, y0)
  * The center may also be computed (centroid or image center)
* Automated test suite:
  * Since version 0.10, DataLab's proxy object has a `toggle_auto_refresh` method to toggle the "Auto-refresh" feature. This feature may be useful to improve performance during the execution of test scripts
  * Test scenarios on signals and images are now using this feature to improve performance
* Signal and image metadata:
  * Added "source" entry to the metadata dictionary, to store the source file path when importing a signal or an image from a file
  * This field is kept while processing the signal/image, in order to keep track of the source file path

📚 Documentation:

* New [Tutorial section](https://datalab-platform.com/en/intro/tutorials/index.html) in the documentation:
  * This section provides a set of tutorials to learn how to use DataLab
  * The following video tutorials are available:
    * Quick demo
    * Adding your own features
  * The following text tutorials are available:
    * Processing a spectrum
    * Detecting blobs on an image
    * Measuring Fabry-Perot fringes
    * Prototyping a custom processing pipeline
* New [API section](https://datalab-platform.com/en/api/index.html) in the documentation:
  * This section explains how to use DataLab as a Python library, by covering the following topics:
    * How to use DataLab algorithms on NumPy arrays
    * How to use DataLab computation features on DataLab objects (signals and images)
    * How to use DataLab I/O features
    * How to use proxy objects to control DataLab remotely
  * This section also provides a complete API reference for DataLab objects and features
  * This fixes [Issue #19](https://github.com/DataLab-Platform/DataLab/issues/19) - Add API documentation (data model, functions on arrays or signal/image objects, ...)

🛠️ Bug fixes:

* Fixed [Issue #29](https://github.com/DataLab-Platform/DataLab/issues/29) - Polynomial fit error: `QDialog [...] argument 1 has an unexpected type 'SignalProcessor'`
* Image ROI extraction feature:
  * Before this release, when extracting a single circular ROI from an image with the "Extract all ROIs into a single image object" option enabled, the result was a single image without the ROI mask (the ROI mask was only available when extracting ROI with the option disabled)
  * This was leading to an unexpected behavior, because one could interpret the result (a square image without the ROI mask) as the result of a single rectangular ROI
  * Now, when extracting a single circular ROI from an image with the "Extract all ROIs into a single image object" option enabled, the result is a single image with the ROI mask (as if the option was disabled)
  * This fixes [Issue #31](https://github.com/DataLab-Platform/DataLab/issues/31) - Single circular ROI extraction: automatically switch to `compute_extract_roi` function
* Analysis on circular ROI:
  * Before this release, when running computations on a circular ROI, the results were unexpected in terms of coordinates (results seemed to be computed in a region located above the actual ROI).
  * This was due to a regression introduced in an earlier release.
  * Now, when defining a circular ROI and running computations on it, the results are computed on the actual ROI
  * This fixes [Issue #33](https://github.com/DataLab-Platform/DataLab/issues/33) - Analysis on circular ROI: unexpected results
* Contour detection on ROI:
  * Before this release, when running contour detection on a ROI, some contours were detected outside the ROI (it may be due to a limitation of the scikit-image `find_contours` function).
  * Now, thanks a workaround, the erroneous contours are filtered out.
  * A new test module `datalab.tests.features.images.contour_fabryperot_app` has been added to test the contour detection feature on a Fabry-Perot image (thanks to [@emarin2642](https://github.com/emarin2642) for the contribution)
  * This fixes [Issue #34](https://github.com/DataLab-Platform/DataLab/issues/34) - Contour detection: unexpected results outside ROI
* Analysis result merging:
  * Before this release, when doing a `1->N` computation (sum, average, product) on a group of signals/images, the analysis results associated to each signal/image were merged into a single result, but only the type of result present in the first signal/image was kept.
  * Now, the analysis results associated to each signal/image are merged into a single result, whatever the type of result is.
* Fixed [Issue #36](https://github.com/DataLab-Platform/DataLab/issues/36) - "Delete all" action enable state is sometimes not refreshed
* Image X/Y swap: when swapping X and Y axes, the regions of interest (ROI) were not removed and not swapped either (ROI are now removed, until we implement the swap feature, if requested)
* "Properties" group box: the "Apply" button was enabled by default, even when no property was modified, which was confusing for the user (the "Apply" button is now disabled by default, and is enabled only when a property is modified)
* Fixed proxy `get_object` method when there is no object to return (`None` is returned instead of an exception)
* Fixed `IndexError: list index out of range` when performing some operations or computations on groups of signals/images (e.g. "ROI extraction", "Peak detection", "Resize", etc.)
* Drag and drop from a file manager: filenames are now sorted alphabetically

## DataLab Version 0.10.1 ##

*Note*: V0.10.0 was almost immediately replaced by V0.10.1 due to a last minute bug fix

💥 New features:

* Features common to signals and images:
  * Added "Real part" and "Imaginary part" features to "Operation" menu
  * Added "Convert data type" feature to "Operation" menu
* Features added following user requests (12/18/2023 meetup @ CEA):
  * Curve and image styles are now saved in the HDF5 file:
    * Curve style covers the following properties: color, line style, line width, marker style, marker size, marker edge color, marker face color, etc.
    * Image style covers the following properties: colormap, interpolation, etc.
    * Those properties were already persistent during the working session, but were lost when saving and reloading the HDF5 file
    * Now, those properties are saved in the HDF5 file and are restored when reloading the HDF5 file
  * New profile extraction features for images:
    * Added "Line profile" to "Operations" menu, to extract a profile from an image along a row or a column
    * Added "Average profile" to "Operations" menu, to extract the average profile on a rectangular area of an image, along a row or a column
  * Image LUT range (contrast/brightness settings) is now saved in the HDF5 file:
    * As for curve and image styles, the LUT range was already persistent during the working session, but was lost when saving and reloading the HDF5 file
    * Now, the LUT range is saved in the HDF5 file and is restored when reloading it
  * Added "Auto-refresh" and "Refresh manually" actions in "View" menu (and main toolbar):
    * When "Auto-refresh" is enabled (default), the plot view is automatically refreshed when a signal/image is modified, added or removed. Even though the refresh is optimized, this may lead to performance issues when working with large datasets.
    * When disabled, the plot view is not automatically refreshed. The user must manually refresh the plot view by clicking on the "Refresh manually" button in the main toolbar or by pressing the standard refresh key (e.g. "F5").
  * Added `toggle_auto_refresh` method to DataLab proxy object:
    * This method allows to toggle the "Auto-refresh" feature from a macro-command, a plugin or a remote control client.
    * A context manager `context_no_refresh` is also available to temporarily disable the "Auto-refresh" feature from a macro-command, a plugin or a remote control client. Typical usage:

      ```python
      with proxy.context_no_refresh():
          # Do something without refreshing the plot view
          proxy.compute_fft() # (...)
      ```

  * Improved curve readability:
    * Until this release, the curve style was automatically set by cycling through
      **PlotPy** predefined styles
    * However, some styles are not suitable for curve readability (e.g. "cyan" and "yellow" colors are not readable on a white background, especially when combined with a "dashed" line style)
    * This release introduces a new curve style management with colors which are distinguishable and accessible, even to color vision deficiency people
* Added "Curve anti-aliasing" feature to "View" menu (and toolbar):
  * This feature allows to enable/disable curve anti-aliasing (default: enabled)
  * When enabled, the curve rendering is smoother but may lead to performance issues when working with large datasets (that's why it can be disabled)
* Added `toggle_show_titles` method to DataLab proxy object. This method allows to toggle the "Show graphical object titles" feature from a macro-command, a plugin or a remote control client.
* Remote client is now checking the server version and shows a warning message if the server version may not be fully compatible with the client version.

🛠️ Bug fixes:

* Image contour detection feature ("Analysis" menu):
  * The contour detection feature was not taking into account the "shape" parameter (circle, ellipse, polygon) when computing the contours. The parameter was stored but really used only when calling the feature a second time.
  * This unintentional behavior led to an `AssertionError` when choosing "polygon" as the contour shape and trying to compute the contours for the first time.
  * This is now fixed (see [Issue #9](https://github.com/DataLab-Platform/DataLab/issues/9) - Image contour detection: `AssertionError` when choosing "polygon" as the contour shape)
* Keyboard shortcuts:
  * The keyboard shortcuts for "New", "Open", "Save", "Duplicate", "Remove", "Delete all" and "Refresh manually" actions were not working properly.
  * Those shortcuts were specific to each signal/image panel, and were working only when the panel on which the shortcut was pressed for the first time was active (when activated from another panel, the shortcut was not working and a warning message was displayed in the console, e.g. `QAction::event: Ambiguous shortcut overload: Ctrl+C`)
  * Besides, the shortcuts were not working at startup (when no panel had focus).
  * This is now fixed: the shortcuts are now working whatever the active panel is, and even at startup (see [Issue #10](https://github.com/DataLab-Platform/DataLab/issues/10) - Keyboard shortcuts not working properly: `QAction::event: Ambiguous shortcut overload: Ctrl+C`)
* "Show graphical object titles" and "Auto-refresh" actions were not working properly:
  * The "Show graphical object titles" and "Auto-refresh" actions were only working on the active signal/image panel, and not on all panels.
  * This is now fixed (see [Issue #11](https://github.com/DataLab-Platform/DataLab/issues/11) - "Show graphical object titles" and "Auto-refresh" actions were working only on current signal/image panel)
* Fixed [Issue #14](https://github.com/DataLab-Platform/DataLab/issues/14) - Saving/Reopening HDF5 project without cleaning-up leads to `ValueError`
* Fixed [Issue #15](https://github.com/DataLab-Platform/DataLab/issues/15) - MacOS: 1. `pip install datalab` error - 2. Missing menus:
  * Part 1: `pip install datalab` error on MacOS was actually an issue from **PlotPy** (see [this issue](https://github.com/PlotPyStack/PlotPy/issues/9)), and has been fixed in PlotPy v2.0.3 with an additional compilation flag indicating to use C++11 standard
  * Part 2: Missing menus on MacOS was due to a PyQt/MacOS bug regarding dynamic menus
* HDF5 file format: when importing an HDF5 dataset as a signal or an image, the dataset attributes were systematically copied to signal/image metadata: we now only copy the attributes which match standard data types (integers, floats, strings) to avoid errors when serializing/deserializing the signal/image object
* Installation/configuration viewer: improved readability (removed syntax highlighting)
* PyInstaller specification file: added missing `skimage` data files manually in order to continue supporting Python 3.8 (see [Issue #12](https://github.com/DataLab-Platform/DataLab/issues/12) - Stand-alone version on Windows 7: missing `api-ms-win-core-path-l1-1-0.dll`)
* Fixed [Issue #13](https://github.com/DataLab-Platform/DataLab/issues/13) - ArchLinux: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found`

## DataLab Version 0.9.2 ##

🛠️ Bug fixes:

* Region of interest (ROI) extraction feature for images:
  * ROI extraction was not working properly when the "Extract all ROIs into a single image object" option was enabled if there was only one defined ROI. The result was an image positioned at the origin (0, 0) instead of the expected position (x0, y0) and the ROI rectangle itself was not removed as expected. This is now fixed (see [Issue #6](https://github.com/DataLab-Platform/DataLab/issues/6) - 'Extract multiple ROI' feature: unexpected result for a single ROI)
  * ROI rectangles with negative coordinates were not properly handled: ROI extraction was raising a `ValueError` exception, and the image mask was not displayed properly. This is now fixed (see [Issue #7](https://github.com/DataLab-Platform/DataLab/issues/7) - Image ROI extraction: `ValueError: zero-size array to reduction operation minimum which has no identity`)
  * ROI extraction was not taking into account the pixel size (dx, dy) and the origin (x0, y0) of the image. This is now fixed (see [Issue #8](https://github.com/DataLab-Platform/DataLab/issues/8) - Image ROI extraction: take into account pixel size)
* Macro-command console is now read-only:
  * The macro-command panel Python console is currently not supporting standard input stream (`stdin`) and this is intended (at least for now)
  * Set Python console read-only to avoid confusion

## DataLab Version 0.9.1 ##

🛠️ Bug fixes:

* French translation is not available on Windows/Stand alone version:
  * Locale was not properly detected on Windows for stand-alone version (frozen with `pyinstaller`) due to an issue with `locale.getlocale()` (function returning `None` instead of the expected locale on frozen applications)
  * This is ultimately a `pyinstaller` issue, but a workaround has been implemented in `guidata` V3.2.2 (see [guidata issue #68](https://github.com/PlotPyStack/guidata/issues/68) - Windows: gettext translation is not working on frozen applications)
  * [Issue #2](https://github.com/DataLab-Platform/DataLab/issues/2) - French translation is not available on Windows Stand alone version
* Saving image to JPEG2000 fails for non integer data:
  * JPEG2000 encoder does not support non integer data or signed integer data
  * Before, DataLab was showing an error message when trying to save incompatible data to JPEG2000: this was not a consistent behavior with other standard image formats (e.g. PNG, JPG, etc.) for which DataLab was automatically converting data to the appropriate format (8-bit unsigned integer)
  * Current behavior is now consistent with other standard image formats: when saving to JPEG2000, DataLab automatically converts data to 8-bit unsigned integer or 16-bit unsigned integer (depending on the original data type)
  * [Issue #3](https://github.com/DataLab-Platform/DataLab/issues/3) - Save image to JPEG2000: 'OSError: encoder error -2 when writing image file'
* Windows stand-alone version shortcuts not showing in current user start menu:
  * When installing DataLab on Windows from a non-administrator account, the shortcuts were not showing in the current user start menu but in the administrator start menu instead (due to the elevated privileges of the installer and the fact that the installer does not support installing shortcuts for all users)
  * Now, the installer *does not* ask for elevated privileges anymore, and shortcuts are installed in the current user start menu (this also means that the current user must have write access to the installation directory)
  * In future releases, the installer will support installing shortcuts for all users if there is a demand for it (see [Issue #5](https://github.com/DataLab-Platform/DataLab/issues/5))
  * [Issue #4](https://github.com/DataLab-Platform/DataLab/issues/4) - Windows: stand-alone version shortcuts not showing in current user start menu
* Installation and configuration window for stand-alone version:
  * Do not show ambiguous error message 'Invalid dependencies' anymore
  * Dependencies are supposed to be checked when building the stand-alone version
* Added PDF documentation to stand-alone version:
  * The PDF documentation was missing in previous release
  * Now, the PDF documentation (in English and French) is included in the stand-alone version

## DataLab Version 0.9.0 ##

New dependencies:

* DataLab is now powered by [PlotPyStack](https://github.com/PlotPyStack):
  * [PythonQwt](https://github.com/PlotPyStack/PythonQwt)
  * [guidata](https://github.com/PlotPyStack/guidata)
  * [PlotPy](https://github.com/PlotPyStack/PlotPy)
* [opencv-python](https://pypi.org/project/opencv-python/) (algorithms for image processing)

New reference platform:

* DataLab is validated on Windows 11 with Python 3.11 and PyQt 5.15
* DataLab is also compatible with other OS (Linux, MacOS) and other Python-Qt bindings and versions (Python 3.8-3.12, PyQt6, PySide6)

New features:

* DataLab is a platform:
  * Added support for plugins
    * Custom processing features available in the "Plugins" menu
    * Custom I/O features: new file formats can be added to the standard I/O features for signals and images
    * Custom HDF5 features: new HDF5 file formats can be added to the standard HDF5 import feature
    * More features to come...
  * Added remote control feature: DataLab can be controlled remotely via a TCP/IP connection (see [Remote control](https://datalab-platform.com/en/remote_control.html))
  * Added macro commands: DataLab can be controlled via a macro file (see [Macro commands](https://datalab-platform.com/en/macro_commands.html))
* General features:
  * Added settings dialog box (see "Settings" entry in "File" menu):
    * General settings
    * Visualization settings
    * Processing settings
    * Etc.
  * New default layout: signal/image panels are on the right side of the main window, visualization panels are on the left side with a vertical toolbar
* Signal/Image features:
  * Added process isolation: each signal/image is processed in a separate process, so that DataLab does not freeze anymore when processing large signals/images
  * Added support for groups: signals and images can be grouped together, and operations can be applied to all objects in a group, or between groups
  * Added warning and error dialogs with detailed traceback links to the source code (warnings may be optionally ignored)
  * Drastically improved performance when selecting objects
  * Optimized performance when showing large images
  * Added support for dropping files on signal/image panel
  * Added "Analysis parameters" group box to show last result input parameters
  * Added "Copy titles to clipboard" feature in "Edit" menu
  * For every single processing feature (operation, processing and analysis menus), the entered parameters (dialog boxes) are stored in cache to be used as defaults the next time the feature is used
* Signal processing:
  * Added support for optional FFT shift (see Settings dialog box)
* Image processing:
  * Added pixel binning operation (X/Y binning factors, operation: sum, mean...)
  * Added "Distribute on a grid" and "Reset image positions" in operation menu
  * Added Butterworth filter
  * Added exposure processing features:
    * Gamma correction
    * Logarithmic correction
    * Sigmoïd correction
  * Added restoration processing features:
    * Total variation denoising filter (TV Chambolle)
    * Bilateral filter (denoising)
    * Wavelet denoising filter
    * White Top-Hat denoising filter
  * Added morphological transforms (disk footprint):
    * White Top-Hat
    * Black Top-Hat
    * Erosion
    * Dilation
    * Opening
    * Closing
  * Added edge detection features:
    * Roberts filter
    * Prewitt filter (vertical, horizontal, both)
    * Sobel filter (vertical, horizontal, both)
    * Scharr filter (vertical, horizontal, both)
    * Farid filter (vertical, horizontal, both)
    * Laplace filter
    * Canny filter
  * Contour detection: added support for polygonal contours (in addition to circle and ellipse contours)
  * Added circle Hough transform (circle detection)
  * Added image intensity levels rescaling
  * Added histogram equalization
  * Added adaptative histogram equalization
  * Added blob detection methods:
    * Difference of Gaussian
    * Determinant of Hessian method
    * Laplacian of Gaussian
    * Blob detection using OpenCV
  * Result shapes and annotations are now transformed (instead of removed) when executing one of the following operations:
    * Rotation (arbitrary angle, +90°, -90°)
    * Symetry (vertical/horizontal)
  * Added support for optional FFT shift (see Settings dialog box)
* Console: added configurable external editor (default: VSCode) to follow the traceback links to the source code
