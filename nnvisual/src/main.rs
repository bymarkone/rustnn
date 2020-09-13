use charts::{Chart, ScaleLinear, ScatterView, MarkerType, Color };
use csv::Reader;
use std::error::Error;
use csv::StringRecord;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn Error>> {
    // Define chart related sizes.
    let mut rdr = Reader::from_path("titanic.csv")?;
    let scatter_data = rdr
        .records()
        .map(|item| item.expect("Unable to read record"))
        .map(|item| to_data(item))
        .map(|(x, y, label)| (f32::from_str(&x).unwrap(), f32::from_str(&y).unwrap(), label))
        .collect();

    let width = 800;
    let height = 600;
    let (top, right, bottom, left) = (90, 40, 50, 60);

    // Create a band scale that will interpolate values in [0, 200] to values in the
    // [0, availableWidth] range (the width of the chart without the margins).
    let x = ScaleLinear::new()
        .set_domain(vec![0.0, 100.0])
        .set_range(vec![0, width - left - right]);

    // Create a linear scale that will interpolate values in [0, 100] range to corresponding
    // values in [availableHeight, 0] range (the height of the chart without the margins).
    // The [availableHeight, 0] range is inverted because SVGs coordinate system's origin is
    // in top left corner, while chart's origin is in bottom left corner, hence we need to invert
    // the range on Y axis for the chart to display as though its origin is at bottom left.
    let y = ScaleLinear::new()
        .set_domain(vec![0.0, 4.0])
        .set_range(vec![height - top - bottom, 0]);

    // You can use your own iterable as data as long as its items implement the `PointDatum` trait.

    // Create Scatter view that is going to represent the data as points.
    let scatter_view = ScatterView::new()
        .set_x_scale(&x)
        .set_y_scale(&y)
        .set_label_visibility(false)
        .set_marker_type(MarkerType::X)
        .set_colors(Color::color_scheme_dark())
        .load_data(&scatter_data).unwrap();

    // Generate and save the chart.
    Chart::new()
        .set_width(width)
        .set_height(height)
        .set_margins(top, right, bottom, left)
        .add_title(String::from("Scatter Chart"))
        .add_view(&scatter_view)
        .add_axis_bottom(&x)
        .add_axis_left(&y)
        .add_left_axis_label("Custom X Axis Label")
        .add_bottom_axis_label("Custom Y Axis Label")
        .save("scatter-chart-multiple-keys.svg").unwrap();

    Ok(())
}

fn to_data(item: StringRecord) -> (String, String, String) {
    (item.get(0).unwrap().to_string(), item.get(1).unwrap().to_string(), item.get(2).unwrap().to_string())
}

