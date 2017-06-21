/*
 A (very) simple UI lib built on top of OpenCV drawing primitives.

 Version: 2.0.0

 Copyright (c) 2016 Fernando Bevilacqua <dovyski@gmail.com>
 Licensed under the MIT license.
*/

#ifndef _CVUI_H_
#define _CVUI_H_

#include <iostream>
#include <stdarg.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

namespace cvui
{
/**
 Initializes the library. You must provide the name of the window where
 the components will be added. It is also possible to tell cvui to handle
 OpenCV's event queue automatically (by informing a value greater than zero
 in the `theDelayWaitKey` parameter of function). In that case, cvui will
 automatically call `cv::waitKey()` within `cvui::update()`, so you don't
 have to worry about it. The value passed to `theDelayWaitKey` will be
 used as the delay for `cv::waitKey()`.
 
 \param theWindowName name of the window where the components will be added
 \param theDelayWaitKey delay value passed to `cv::waitKey()`. If a negative value is informed (default is `-1`), cvui will not automatically call `cv::waitKey()` within `cvui::update()`, which will disable keyboard shortcuts for all components. If you want to enable keyboard shortcut for components (e.g. using & in a button label), you must specify a positive value for this param.
*/
void init(const cv::String& theWindowName, int theDelayWaitKey = -1);

/**
 Return the last key that was pressed. This function will only
 work if a value greater than zero was passed to `cvui::init()`
 as the delay waitkey parameter.

 \sa init()
 */
int lastKeyPressed();

/**
 Display a button. The size of the button will be automatically adjusted to
 properly house the label content.

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theLabel text displayed inside the button.
 \return `true` everytime the user clicks the button.
*/
bool button(cv::Mat& theWhere, int theX, int theY, const cv::String& theLabel);

/**
 Display a button. The button size will be defined by the width and height parameters,
 no matter the content of the label.

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theWidth width of the button.
 \param theHeight height of the button.
 \param theLabel text displayed inside the button.
 \return `true` everytime the user clicks the button.
*/
bool button(cv::Mat& theWhere, int theX, int theY, int theWidth, int theHeight, const cv::String& theLabel);

/**
 Display a button whose graphics are images (cv::Mat). The button accepts three images to describe its states,
 which are idle (no mouse interaction), over (mouse is over the button) and down (mouse clicked the button).
 The button size will be defined by the width and height of the images. 

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theIdle an image that will be rendered when the button is not interacting with the mouse cursor.
 \param theOver an image that will be rendered when the mouse cursor is over the button.
 \param theDown an image that will be rendered when the mouse cursor clicked the button (or is clicking).
 \return `true` everytime the user clicks the button.

 \sa button()
 \sa image()
 \sa iarea()
*/
bool button(cv::Mat& theWhere, int theX, int theY, cv::Mat& theIdle, cv::Mat& theOver, cv::Mat& theDown);

/**
 Display an image (cv::Mat). 

 \param theWhere the image/frame where the provded image should be rendered.
 \param theX position X where the image should be placed.
 \param theY position Y where the image should be placed.
 \param theImage an image to be rendered in the specified destination.

 \sa button()
 \sa iarea()
*/
void image(cv::Mat& theWhere, int theX, int theY, cv::Mat& theImage);

/**
 Display a checkbox. You can use the state parameter to monitor if the
 checkbox is checked or not.

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theLabel text displayed besides the clickable checkbox square.
 \param theState describes the current state of the checkbox: `true` means the checkbox is checked.
 \param theColor color of the label in the format `0xRRGGBB`, e.g. `0xff0000` for red.
 \return a boolean value that indicates the current state of the checkbox, `true` if it is checked.
*/
bool checkbox(cv::Mat& theWhere, int theX, int theY, const cv::String& theLabel, bool *theState, unsigned int theColor = 0xCECECE);

/**
 Display a piece of text.

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theText the text content.
 \param theFontScale size of the text.
 \param theColor color of the text in the format `0xRRGGBB`, e.g. `0xff0000` for red.

 \sa printf()
*/
void text(cv::Mat& theWhere, int theX, int theY, const cv::String& theText, double theFontScale = 0.4, unsigned int theColor = 0xCECECE);

/**
 Display a piece of text that can be formated using `stdio's printf()` style. For instance
 if you want to display text mixed with numbers, you can use:

 ```
 printf(frame, 10, 15, 0.4, 0xff0000, "Text: %d and %f", 7, 3.1415);
 ```

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theFontScale size of the text.
 \param theColor color of the text in the format `0xRRGGBB`, e.g. `0xff0000` for red.
 \param theFmt formating string as it would be supplied for `stdio's printf()`, e.g. `"Text: %d and %f", 7, 3.1415`.
 
 \sa text()
*/
void printf(cv::Mat& theWhere, int theX, int theY, double theFontScale, unsigned int theColor, const char *theFmt, ...);

/**
 Display a piece of text that can be formated using `stdio's printf()` style. For instance
 if you want to display text mixed with numbers, you can use:

 ```
 printf(frame, 10, 15, 0.4, 0xff0000, "Text: %d and %f", 7, 3.1415);
 ```

 The size and color of the text will be based on cvui's default values.

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theFmt formating string as it would be supplied for `stdio's printf()`, e.g. `"Text: %d and %f", 7, 3.1415`.

 \sa text()
*/
void printf(cv::Mat& theWhere, int theX, int theY, const char *theFmt, ...);

/**
 Display a counter for integer values that the user can increase/descrease
 by clicking the up and down arrows.

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theValue the current value of the counter.
 \param theStep the amount that should be increased/decreased when the user interacts with the counter buttons
 \param theFormat how the value of the counter should be presented, as it was printed by `stdio's printf()`. E.g. `"%d"` means the value will be displayed as an integer, `"%0d"` integer with one leading zero, etc.
 \return an integer that corresponds to the current value of the counter.
*/
int counter(cv::Mat& theWhere, int theX, int theY, int *theValue, int theStep = 1, const char *theFormat = "%d");

/**
 Display a counter for float values that the user can increase/descrease
 by clicking the up and down arrows.

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theValue the current value of the counter.
 \param theStep the amount that should be increased/decreased when the user interacts with the counter buttons
 \param theFormat how the value of the counter should be presented, as it was printed by `stdio's printf()`. E.g. `"%f"` means the value will be displayed as a regular float, `"%.2f"` float with two digits after the point, etc.
 \return a float that corresponds to the current value of the counter.
*/
double counter(cv::Mat& theWhere, int theX, int theY, double *theValue, double theStep = 0.5, const char *theFormat = "%.2f");

/**
 Display a trackbar for numeric values that the user can increase/decrease
 by clicking and/or dragging the marker right or left. This component uses templates
 so it is imperative that you make it very explicit the type of `theValue`, `theMin`, `theMax` and `theStep`,
 otherwise you might end up with weird compilation errors. 
 
 Example:

 ```
 // using double
 trackbar(where, x, y, width, &doubleValue, 0.0, 50.0);

 // using float
 trackbar(where, x, y, width, &floatValue, 0.0f, 50.0f);

 // using char
 trackbar(where, x, y, width, &charValue, (char)1, (char)10);
 ```

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theWidth the width of the trackbar.
 \param theValue the current value of the trackbar. It will be modified when the user interacts with the trackbar. Any numeric type can be used, e.g. float, double, long double, int, char, uchar.
 \param theMin the minimum value allowed for the trackbar.
 \param theMax the maximum value allowed for the trackbar.
 \param theSegments number of segments the trackbar will have (default is 1). Segments can be seen as groups of numbers in the scale of the trackbar. For example, 1 segment means a single groups of values (no extra labels along the scale), 2 segments mean the trackbar values will be divided in two groups and a label will be placed at the middle of the scale.
 \param theLabelFormat formating string that will be used to render the labels, e.g. `%.2Lf` (Lf *not lf). No matter the type of the `theValue` param, internally trackbar stores it as a `long double`, so the formating string will *always* receive a `long double` value to format. If you are using a trackbar with integers values, for instance, you can supress decimals using a formating string such as `%.0Lf` to format your labels.
 \param theOptions options to customize the behavior/appearance of the trackbar, expressed as a bitset. Available options are defined as `TRACKBAR_` constants and they can be combined using the bitwise `|` operand. Available options are: `TRACKBAR_HIDE_SEGMENT_LABELS` (do not render segment labels, but do render min/max labels), `TRACKBAR_HIDE_STEP_SCALE` (do not render the small lines indicating values in the scale), `TRACKBAR_DISCRETE` (changes of the trackbar value are multiples of theDiscreteStep param), `TRACKBAR_HIDE_MIN_MAX_LABELS` (do not render min/max labels), `TRACKBAR_HIDE_VALUE_LABEL` (do not render the current value of the trackbar below the moving marker), `TRACKBAR_HIDE_LABELS` (do not render labels at all).
 \param theDiscreteStep the amount that the trackbar marker will increase/decrease when the marker is dragged right/left (if option TRACKBAR_DISCRETE is ON)
 \return `true` when the value of the trackbar changed.

 \sa counter()
*/
template <typename T>
bool trackbar(cv::Mat& theWhere, int theX, int theY, int theWidth, T *theValue, T theMin, T theMax, int theSegments = 1, const char *theLabelFormat = "%.1Lf", unsigned int theOptions = 0, T theDiscreteStep = 1);

/**
 Display a window (a block with a title and a body).

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theWidth width of the window.
 \param theHeight height of the window.
 \param theTitle text displayed as the title of the window.

 \sa rect()
*/
void window(cv::Mat& theWhere, int theX, int theY, int theWidth, int theHeight, const cv::String& theTitle);

/**
 Display a filled rectangle.

 \param theWhere the image/frame where the component should be rendered.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theWidth width of the rectangle.
 \param theHeight height of the rectangle.
 \param theBorderColor color of rectangle's border in the format `0xRRGGBB`, e.g. `0xff0000` for red.
 \param theFillingColor color of rectangle's filling in the format `0xAARRGGBB`, e.g. `0x00ff0000` for red, `0xff000000` for transparent filling.

 \sa image()
*/
void rect(cv::Mat& theWhere, int theX, int theY, int theWidth, int theHeight, unsigned int theBorderColor, unsigned int theFillingColor = 0xff000000);

/**
 Display the values of a vector as a sparkline.

 \param theWhere the image/frame where the component should be rendered.
 \param theValues a vector containing the values to be used in the sparkline.
 \param theX position X where the component should be placed.
 \param theY position Y where the component should be placed.
 \param theWidth width of the sparkline.
 \param theHeight height of the sparkline.
 \param theColor color of sparkline in the format `0xRRGGBB`, e.g. `0xff0000` for red.

 \sa trackbar()
*/
void sparkline(cv::Mat& theWhere, std::vector<double>& theValues, int theX, int theY, int theWidth, int theHeight, unsigned int theColor = 0x00FF00);

/**
 Create an interaction area that reports activity with the mouse cursor.
 The tracked interactions are returned by the function and they are:

 `OUT` when the cursor is not over the iarea.
 `OVER` when the cursor is over the iarea.
 `DOWN` when the cursor is pressed over the iarea, but not released yet.
 `CLICK` when the cursor clicked (pressed and released) within the iarea.

 This function creates no visual output on the screen. It is intended to
 be used as an auxiliary tool to create interactions.

 \param theX position X where the interactive area should be placed.
 \param theY position Y where the interactive area should be placed.
 \param theWidth width of the interactive area.
 \param theHeight height of the interactive area.
 \return an integer value representing the current state of interaction with the mouse cursor. It can be `OUT` (cursor is not over the area), `OVER` (cursor is over the area), `DOWN` (cursor is pressed over the area, but not released yet) and `CLICK` (cursor clicked, i.e. pressed and released, within the area).

 \sa button()
 \sa image()
*/
int iarea(int theX, int theY, int theWidth, int theHeight);

/**
 Start a new row.
 
 One of the most annoying tasks when building UI is to calculate 
 where each component should be placed on the screen. cvui has
 a set of methods that abstract the process of positioning
 components, so you don't have to think about assigning a
 X and Y coordinate. Instead you just add components and cvui
 will place them as you go.

 You use `beginRow()` to start a group of elements. After `beginRow()`
 has been called, all subsequent component calls don't have to specify
 the frame where the component should be rendered nor its position.
 The position of the component will be automatically calculated by cvui
 based on the components within the group. All components are placed
 side by side, from left to right.

 E.g.

 ```
 beginRow(frame, x, y, width, height);
  text("test");
  button("btn");
 endRow();
 ```

 Rows and columns can be nested, so you can create columns/rows within
 columns/rows as much as you want. It's important to notice that any
 component within `beginRow()` and `endRow()` *do not* specify the position
 where the component is rendered, which is also true for `beginRow()`.
 As a consequence, **be sure you are calling `beginRow(width, height)`
 when the call is nested instead of `beginRow(x, y, width, height)`**,
 otherwise cvui will throw an error.

 E.g.

 ```
 beginRow(frame, x, y, width, height);
  text("test");       
  button("btn"); 

  beginColumn();      // no frame nor x,y parameters here!
   text("column1");
   text("column2");
  endColumn();
 endRow();
 ```

 Don't forget to call `endRow()` to finish the row, otherwise cvui will throw an error.

 \param theWhere the image/frame where the components within this block should be rendered.
 \param theX position X where the row should be placed.
 \param theY position Y where the row should be placed.
 \param theWidth width of the row. If a negative value is specified, the width of the row will be automatically calculated based on the content of the block.
 \param theHeight height of the row. If a negative value is specified, the height of the row will be automatically calculated based on the content of the block.
 \param thePadding space, in pixels, among the components of the block.

 \sa beginColumn()
 \sa endRow()
 \sa endColumn()
*/
void beginRow(cv::Mat &theWhere, int theX, int theY, int theWidth = -1, int theHeight = -1, int thePadding = 0);

/**
 Ends a row. You must call this function only if you have previously called
 its counter part, the `beginRow()` function. 

\sa beginRow()
\sa beginColumn()
\sa endColumn()
*/
void endRow();

/**
 Start a new column.

 One of the most annoying tasks when building UI is to calculate
 where each component should be placed on the screen. cvui has
 a set of methods that abstract the process of positioning
 components, so you don't have to think about assigning a
 X and Y coordinate. Instead you just add components and cvui
 will place them as you go.

 You use `beginColumn()` to start a group of elements. After `beginColumn()`
 has been called, all subsequent component calls don't have to specify
 the frame where the component should be rendered nor its position.
 The position of the component will be automatically calculated by cvui
 based on the components within the group. All components are placed
 below each other, from the top of the screen towards the bottom.

 E.g.

 ```
 beginColumn(frame, x, y, width, height);
  text("test");
  button("btn");
 endColumn();
 ```

 Rows and columns can be nested, so you can create columns/rows within
 columns/rows as much as you want. It's important to notice that any
 component within `beginColumn()` and `endColumn()` *do not* specify the position
 where the component is rendered, which is also true for `beginColumn()`.
 As a consequence, **be sure you are calling `beginColumn(width, height)`
 when the call is nested instead of `beginColumn(x, y, width, height)`**,
 otherwise cvui will throw an error.

E.g.

```
beginColumn(frame, x, y, width, height);
 text("test");
 button("btn");

 beginRow();      // no frame nor x,y parameters here!
  text("column1");
  text("column2");
 endRow();
endColumn();
```

Don't forget to call `endColumn()` to finish the column, otherwise cvui will throw an error.

\param theWhere the image/frame where the components within this block should be rendered.
\param theX position X where the row should be placed.
\param theY position Y where the row should be placed.
\param theWidth width of the column. If a negative value is specified, the width of the column will be automatically calculated based on the content of the block.
\param theHeight height of the column. If a negative value is specified, the height of the column will be automatically calculated based on the content of the block.
\param thePadding space, in pixels, among the components of the block.

\sa beginRow()
\sa endColumn()
\sa endRow()
*/
void beginColumn(cv::Mat &theWhere, int theX, int theY, int theWidth = -1, int theHeight = -1, int thePadding = 0);

/**
 Ends a column. You must call this function only if you have previously called
 its counter part, the `beginColumn()` function.

 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
*/
void endColumn();

/**
Starts a row. This function behaves in the same way as `beginRow(frame, x, y, width, height)`,
however it is suposed to be used within `begin*()/end*()` blocks since they require components
not to inform frame nor x,y coordinates.

\sa beginColumn()
\sa endRow()
\sa endColumn()
*/
void beginRow(int theWidth = -1, int theHeight = -1, int thePadding = 0);

/**
Starts a column. This function behaves in the same way as `beginColumn(frame, x, y, width, height)`,
however it is suposed to be used within `begin*()/end*()` blocks since they require components
not to inform frame nor x,y coordinates.

\sa beginColumn()
\sa endRow()
\sa endColumn()
*/
void beginColumn(int theWidth = -1, int theHeight = -1, int thePadding = 0);

/**
 Adds an arbitrary amount of space between components within a `begin*()` and `end*()` block.
 The function is aware of context, so if it is used within a `beginColumn()` and
 `endColumn()` block, the space will be vertical. If it is used within a `beginRow()`
 and `endRow()` block, space will be horizontal.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theValue the amount of space to be added.

 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
void space(int theValue = 5);

/**
 Display a piece of text within a `begin*()` and `end*()` block.
 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theText the text content.
 \param theFontScale size of the text.
 \param theColor color of the text in the format `0xRRGGBB`, e.g. `0xff0000` for red.

 \sa printf()
 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
void text(const cv::String& theText, double theFontScale = 0.4, unsigned int theColor = 0xCECECE);

// 
/**
 Display a button within a `begin*()` and `end*()` block.
 The button size will be defined by the width and height parameters,
 no matter the content of the label.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theWidth width of the button.
 \param theHeight height of the button.
 \param theLabel text displayed inside the button. You can set shortcuts by pre-pending them with "&"
 \return `true` everytime the user clicks the button.

 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
bool button(int theWidth, int theHeight, const cv::String& theLabel);

/**
 Display a button within a `begin*()` and `end*()` block. The size of the button will be
 automatically adjusted to properly house the label content.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theLabel text displayed inside the button. You can set shortcuts by pre-pending them with "&"
 \return `true` everytime the user clicks the button.

 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
bool button(const cv::String& theLabel);

/**
 Display a button whose graphics are images (cv::Mat).

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 The button accepts three images to describe its states,
 which are idle (no mouse interaction), over (mouse is over the button) and down (mouse clicked the button).
 The button size will be defined by the width and height of the images.

 \param theIdle an image that will be rendered when the button is not interacting with the mouse cursor.
 \param theOver an image that will be rendered when the mouse cursor is over the button.
 \param theDown an image that will be rendered when the mouse cursor clicked the button (or is clicking).
 \return `true` everytime the user clicks the button.

 \sa button()
 \sa image()
 \sa iarea()
 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
bool button(cv::Mat& theIdle, cv::Mat& theOver, cv::Mat& theDown);

/**
 Display an image (cv::Mat) within a `begin*()` and `end*()` block

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theWhere the image/frame where the provded image should be rendered.
 \param theX position X where the image should be placed.
 \param theY position Y where the image should be placed.
 \param theImage an image to be rendered in the specified destination.

 \sa button()
 \sa iarea()
 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
void image(cv::Mat& theImage);

/**
 Display a checkbox within a `begin*()` and `end*()` block. You can use the state parameter
 to monitor if the checkbox is checked or not.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theLabel text displayed besides the clickable checkbox square.
 \param theState describes the current state of the checkbox: `true` means the checkbox is checked.
 \param theColor color of the label in the format `0xRRGGBB`, e.g. `0xff0000` for red.
 \return a boolean value that indicates the current state of the checkbox, `true` if it is checked.

 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
bool checkbox(const cv::String& theLabel, bool *theState, unsigned int theColor = 0xCECECE);

/**
 Display a piece of text within a `begin*()` and `end*()` block.
 
 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 The text can be formated using `stdio's printf()` style. For instance if you want to display text mixed
 with numbers, you can use:

 ```
 printf(0.4, 0xff0000, "Text: %d and %f", 7, 3.1415);
 ```

\param theFontScale size of the text.
\param theColor color of the text in the format `0xRRGGBB`, e.g. `0xff0000` for red.
\param theFmt formating string as it would be supplied for `stdio's printf()`, e.g. `"Text: %d and %f", 7, 3.1415`.

\sa text()
\sa beginColumn()
\sa beginRow()
\sa endRow()
\sa endColumn()
*/
void printf(double theFontScale, unsigned int theColor, const char *theFmt, ...);

/**
 Display a piece of text that can be formated using `stdio's printf()` style.
 
 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 For instance if you want to display text mixed with numbers, you can use:

 ```
 printf(frame, 10, 15, 0.4, 0xff0000, "Text: %d and %f", 7, 3.1415);
 ```

 The size and color of the text will be based on cvui's default values.

 \param theFmt formating string as it would be supplied for `stdio's printf()`, e.g. `"Text: %d and %f", 7, 3.1415`.

 \sa text()
 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
void printf(const char *theFmt, ...);

/**
 Display a counter for integer values that the user can increase/descrease
 by clicking the up and down arrows.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theValue the current value of the counter.
 \param theStep the amount that should be increased/decreased when the user interacts with the counter buttons.
 \param theFormat how the value of the counter should be presented, as it was printed by `stdio's printf()`. E.g. `"%d"` means the value will be displayed as an integer, `"%0d"` integer with one leading zero, etc.
 \return an integer that corresponds to the current value of the counter.

\sa printf()
\sa beginColumn()
\sa beginRow()
\sa endRow()
\sa endColumn()
*/
int counter(int *theValue, int theStep = 1, const char *theFormat = "%d");

/**
 Display a counter for float values that the user can increase/descrease
 by clicking the up and down arrows.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theValue the current value of the counter.
 \param theStep the amount that should be increased/decreased when the user interacts with the counter buttons.
 \param theFormat how the value of the counter should be presented, as it was printed by `stdio's printf()`. E.g. `"%d"` means the value will be displayed as an integer, `"%0d"` integer with one leading zero, etc.
 \return an float that corresponds to the current value of the counter.

 \sa printf()
 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
double counter(double *theValue, double theStep = 0.5, const char *theFormat = "%.2f");

/**
 Display a trackbar for numeric values that the user can increase/decrease
 by clicking and/or dragging the marker right or left.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 This component uses templates so it is imperative that you make it very explicit
 the type of `theValue`, `theMin`, `theMax` and `theStep`, otherwise you might end up with
 weird compilation errors.

 Example:

 ```
 // using double
 trackbar(width, &doubleValue, 0.0, 50.0);

 // using float
 trackbar(width, &floatValue, 0.0f, 50.0f);

 // using char
 trackbar(width, &charValue, (char)1, (char)10);
 ```

 \param theWidth the width of the trackbar.
 \param theValue the current value of the trackbar. It will be modified when the user interacts with the trackbar. Any numeric type can be used, e.g. float, double, long double, int, char, uchar.
 \param theMin the minimum value allowed for the trackbar.
 \param theMax the maximum value allowed for the trackbar.
 \param theSegments number of segments the trackbar will have (default is 1). Segments can be seen as groups of numbers in the scale of the trackbar. For example, 1 segment means a single groups of values (no extra labels along the scale), 2 segments mean the trackbar values will be divided in two groups and a label will be placed at the middle of the scale.
 \param theLabelFormat formating string that will be used to render the labels, e.g. `%.2Lf`. No matter the type of the `theValue` param, internally trackbar stores it as a `long double`, so the formating string will *always* receive a `long double` value to format. If you are using a trackbar with integers values, for instance, you can supress decimals using a formating string as `%.0Lf` to format your labels.
 \param theOptions options to customize the behavior/appearance of the trackbar, expressed as a bitset. Available options are defined as `TRACKBAR_` constants and they can be combined using the bitwise `|` operand. Available options are: `TRACKBAR_HIDE_SEGMENT_LABELS` (do not render segment labels, but do render min/max labels), `TRACKBAR_HIDE_STEP_SCALE` (do not render the small lines indicating values in the scale), `TRACKBAR_DISCRETE` (changes of the trackbar value are multiples of informed step param), `TRACKBAR_HIDE_MIN_MAX_LABELS` (do not render min/max labels), `TRACKBAR_HIDE_VALUE_LABEL` (do not render the current value of the trackbar below the moving marker), `TRACKBAR_HIDE_LABELS` (do not render labels at all).
 \param theDiscreteStep the amount that the trackbar marker will increase/decrease when the marker is dragged right/left (if option TRACKBAR_DISCRETE is ON)
 \return `true` when the value of the trackbar changed.

 \sa counter()
 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
template <typename T> // T can be any float type (float, double, long double)
bool trackbar(int theWidth, T *theValue, T theMin, T theMax, int theSegments = 1, const char *theLabelFormat = "%.1Lf", unsigned int theOptions = 0, T theDiscreteStep = 1);

/**
 Display a window (a block with a title and a body) within a `begin*()` and `end*()` block.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theWidth width of the window.
 \param theHeight height of the window.
 \param theTitle text displayed as the title of the window.

 \sa rect()
 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
void window(int theWidth, int theHeight, const cv::String& theTitle);

/**
 Display a rectangle within a `begin*()` and `end*()` block.
 
 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theWidth width of the rectangle.
 \param theHeight height of the rectangle.
 \param theBorderColor color of rectangle's border in the format `0xRRGGBB`, e.g. `0xff0000` for red.
 \param theFillingColor color of rectangle's filling in the format `0xAARRGGBB`, e.g. `0x00ff0000` for red, `0xff000000` for transparent filling.

 \sa window()
 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
void rect(int theWidth, int theHeight, unsigned int theBorderColor, unsigned int theFillingColor = 0xff000000);

/**
 Display the values of a vector as a sparkline within a `begin*()` and `end*()` block.

 IMPORTANT: this function can only be used within a `begin*()/end*()` block, otherwise it does nothing.

 \param theValues vector with the values that will be rendered as a sparkline.
 \param theWidth width of the sparkline.
 \param theHeight height of the sparkline.
 \param theColor color of sparkline in the format `0xRRGGBB`, e.g. `0xff0000` for red.

 \sa beginColumn()
 \sa beginRow()
 \sa endRow()
 \sa endColumn()
*/
void sparkline(std::vector<double>& theValues, int theWidth, int theHeight, unsigned int theColor = 0x00FF00);

/**
 Updates the library internal things. You need to call this function **AFTER** you are done adding/manipulating
 UI elements in order for them to react to mouse interactions.
*/
void update();

// Internally used to handle mouse events
void handleMouse(int theEvent, int theX, int theY, int theFlags, void* theData);

// Compatibility macros to allow compilation with either OpenCV 2.x or OpenCV 3.x
#if (CV_MAJOR_VERSION < 3)
	#define CVUI_ANTIALISED CV_AA
#else
	#define CVUI_ANTIALISED cv::LINE_AA
#endif
#define CVUI_FILLED -1

// Check for Windows-specific functions and react accordingly
#if !defined(_MSC_VER)
	#define vsprintf_s vsprintf
	#define sprintf_s sprintf
#endif

// Check for Unix stuff
#ifdef __GNUC__
	// just to remove the warning under gcc that is introduced by the VERSION variable below
	// (needed for those who compile with -Werror (make warning as errors)
	#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// Lib version
static const char *VERSION = "2.0.0";

const int ROW = 0;
const int COLUMN = 1;
const int DOWN = 2;
const int CLICK = 3;
const int OVER = 4;
const int OUT = 5;

// Constants regarding components
const unsigned int TRACKBAR_HIDE_SEGMENT_LABELS = 1;
const unsigned int TRACKBAR_HIDE_STEP_SCALE = 2;
const unsigned int TRACKBAR_DISCRETE = 4;
const unsigned int TRACKBAR_HIDE_MIN_MAX_LABELS = 8;
const unsigned int TRACKBAR_HIDE_VALUE_LABEL = 16;
const unsigned int TRACKBAR_HIDE_LABELS = 32;

// Describes the block structure used by the lib to handle `begin*()` and `end*()` calls.
typedef struct {
	cv::Mat where;			// where the block should be rendered to.
	cv::Rect rect;			// the size and position of the block.
	cv::Rect fill;			// the filled area occuppied by the block as it gets modified by its inner components.
	cv::Point anchor;		// the point where the next component of the block should be rendered.
	int padding;			// padding among components within this block.
	int type;				// type of the block, e.g. ROW or COLUMN.
} cvui_block_t;

// Describes a component label, including info about a shortcut.
// If a label contains "Re&start", then:
// - hasShortcut will be true
// - shortcut will be 's'
// - textBeforeShortcut will be "Re"
// - textAfterShortcut will be "tart"
typedef struct {
	bool hasShortcut;
	char shortcut;
	std::string textBeforeShortcut;
	std::string textAfterShortcut;
} cvui_label_t;

// Internal namespace with all code that is shared among components/functions.
// You should probably not be using anything from here.
namespace internal
{
	// Variables to keep track of mouse events and stuff
	static bool gMouseJustReleased = false;
	static bool gMousePressed = false;
	static cv::Point gMouse;
	static char gBuffer[1024];
	static int gLastKeyPressed;
	static int gDelayWaitKey;
	static cvui_block_t gScreen;

	struct TrackbarParams {
		long double min;
		long double max;
		long double step;
		int segments;
		unsigned int options;
		std::string labelFormat;

		inline TrackbarParams()
			: min(0.)
			, max(25.)
			, step(1.)
			, segments(0)
			, options(0)
			, labelFormat("%.0Lf")
		{}
	};

	static cvui_block_t gStack[100]; // TODO: make it dynamic?
	static int gStackCount = -1;
	static const int gTrackbarMarginX = 14;

	bool bitsetHas(unsigned int theBitset, unsigned int theValue);
	void error(int theId, std::string theMessage);
	void updateLayoutFlow(cvui_block_t& theBlock, cv::Size theSize);
	bool blockStackEmpty();
	cvui_block_t& topBlock();
	cvui_block_t& pushBlock();
	cvui_block_t& popBlock();
	void begin(int theType, cv::Mat &theWhere, int theX, int theY, int theWidth, int theHeight, int thePadding);
	void end(int theType);
	cvui_label_t createLabel(const std::string &theLabel);
	int iarea(int theX, int theY, int theWidth, int theHeight);
	bool button(cvui_block_t& theBlock, int theX, int theY, int theWidth, int theHeight, const cv::String& theLabel, bool theUpdateLayout);
	bool button(cvui_block_t& theBlock, int theX, int theY, const cv::String& theLabel);
	bool button(cvui_block_t& theBlock, int theX, int theY, cv::Mat& theIdle, cv::Mat& theOver, cv::Mat& theDown, bool theUpdateLayout);
	void image(cvui_block_t& theBlock, int theX, int theY, cv::Mat& theImage);
	bool checkbox(cvui_block_t& theBlock, int theX, int theY, const cv::String& theLabel, bool *theState, unsigned int theColor);
	void text(cvui_block_t& theBlock, int theX, int theY, const cv::String& theText, double theFontScale, unsigned int theColor, bool theUpdateLayout);
	int counter(cvui_block_t& theBlock, int theX, int theY, int *theValue, int theStep, const char *theFormat);
	double counter(cvui_block_t& theBlock, int theX, int theY, double *theValue, double theStep, const char *theFormat);
	void window(cvui_block_t& theBlock, int theX, int theY, int theWidth, int theHeight, const cv::String& theTitle);
	void rect(cvui_block_t& theBlock, int theX, int theY, int theWidth, int theHeight, unsigned int theBorderColor, unsigned int theFillingColor);
	void sparkline(cvui_block_t& theBlock, std::vector<double>& theValues, int theX, int theY, int theWidth, int theHeight, unsigned int theColor);
	bool trackbar(cvui_block_t &theBlock, int theX, int theY, int theWidth, long double *theValue, const TrackbarParams& theParams);
	inline void trackbarForceValuesAsMultiplesOfSmallStep(const TrackbarParams & theParams, long double *theValue);
	inline long double trackbarXPixelToValue(const TrackbarParams & theParams, cv::Rect & theBounding, int thePixelX);
	inline int trackbarValueToXPixel(const TrackbarParams & theParams, cv::Rect & theBounding, long double theValue);
	inline double clamp01(double value);
	void findMinMax(std::vector<double>& theValues, double *theMin, double *theMax);
	cv::Scalar hexToScalar(unsigned int theColor);
	void resetRenderingBuffer(cvui_block_t& theScreen);

	template <typename T> // T can be any floating point type (float, double, long double)
	TrackbarParams makeTrackbarParams(T min, T max, int theDecimals = 1, int theSegments = 1, T theStep = -1., unsigned int theOptions = 0, const char *theFormat = "%.1Lf");

	template<typename T>
	bool trackbar(T *theValue, const TrackbarParams& theParams);

	template <typename T> // T can be any numeric type (int, double, unsigned int, etc)
	bool trackbar(cv::Mat& theWhere, int theX, int theY, int theWidth, T *theValue, const TrackbarParams& theParams);

	template<typename num_type>
	TrackbarParams makeTrackbarParams(num_type theMin, num_type theMax, num_type theStep, int theSegments, const char *theLabelFormat, unsigned int theOptions) {
		TrackbarParams aParams;

		aParams.min = (long double)theMin;
		aParams.max = (long double)theMax;
		aParams.step = (long double)theStep;
		aParams.options = theOptions;
		aParams.segments = theSegments;
		aParams.labelFormat = theLabelFormat;
	
		return aParams;
	}

	template <typename num_type>
	bool trackbar(int theWidth, num_type *theValue, const TrackbarParams& theParams) {
		cvui_block_t& aBlock = internal::topBlock();

		long double aValueAsDouble = static_cast<long double>(*theValue);
		bool aResult = internal::trackbar(aBlock, aBlock.anchor.x, aBlock.anchor.y, theWidth, &aValueAsDouble, theParams);
		*theValue = static_cast<num_type>(aValueAsDouble);

		return aResult;
	}

	template <typename num_type>
	bool trackbar(cv::Mat& theWhere, int theX, int theY, int theWidth, num_type *theValue, const TrackbarParams& theParams) {
		gScreen.where = theWhere;

		long double aValueAsDouble = static_cast<long double>(*theValue);
		bool aResult = internal::trackbar(gScreen, theX, theY, theWidth, &aValueAsDouble, theParams);
		*theValue = static_cast<num_type>(aValueAsDouble);
		
		return aResult;
	}
}

// Internal namespace that contains all rendering functions.
namespace render {
	void text(cvui_block_t& theBlock, const cv::String& theText, cv::Point& thePos, double theFontScale, unsigned int theColor);
	void button(cvui_block_t& theBlock, int theState, cv::Rect& theShape, const cv::String& theLabel);
	void buttonLabel(cvui_block_t& theBlock, int theState, cv::Rect theRect, const cv::String& theLabel, cv::Size& theTextSize);
	void image(cvui_block_t& theBlock, cv::Rect& theRect, cv::Mat& theImage);
	void counter(cvui_block_t& theBlock, cv::Rect& theShape, const cv::String& theValue);
	void trackbarHandle(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams, cv::Rect& theWorkingArea);
	void trackbarPath(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams, cv::Rect& theWorkingArea);
	void trackbarSteps(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams, cv::Rect& theWorkingArea);
	void trackbarSegmentLabel(cvui_block_t& theBlock, cv::Rect& theShape, const internal::TrackbarParams &theParams, long double theValue, cv::Rect& theWorkingArea, bool theShowLabel);
	void trackbarSegments(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams, cv::Rect& theWorkingArea);
	void trackbar(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams);
	void checkbox(cvui_block_t& theBlock, int theState, cv::Rect& theShape);
	void checkboxLabel(cvui_block_t& theBlock, cv::Rect& theRect, const cv::String& theLabel, cv::Size& theTextSize, unsigned int theColor);
	void checkboxCheck(cvui_block_t& theBlock, cv::Rect& theShape);
	void window(cvui_block_t& theBlock, cv::Rect& theTitleBar, cv::Rect& theContent, const cv::String& theTitle);
	void rect(cvui_block_t& theBlock, cv::Rect& thePos, unsigned int theBorderColor, unsigned int theFillingColor);
	void sparkline(cvui_block_t& theBlock, std::vector<double>& theValues, cv::Rect &theRect, double theMin, double theMax, unsigned int theColor);

	int putText(cvui_block_t& theBlock, int theState, cv::Scalar aColor, const std::string& theText, const cv::Point & thePosition);
	int putTextCentered(cvui_block_t& theBlock, const cv::Point & position, const std::string &text);
}

template <typename num_type>
bool trackbar(cv::Mat& theWhere, int theX, int theY, int theWidth, num_type *theValue, num_type theMin, num_type theMax, int theSegments, const char *theLabelFormat, unsigned int theOptions, num_type theDiscreteStep) {
	internal::TrackbarParams aParams = internal::makeTrackbarParams(theMin, theMax, theDiscreteStep, theSegments, theLabelFormat, theOptions);
	return trackbar<num_type>(theWhere, theX, theY, theWidth, theValue, aParams);
}

template <typename num_type>
bool trackbar(int theWidth, num_type *theValue, num_type theMin, num_type theMax, int theSegments, const char *theLabelFormat, unsigned int theOptions, num_type theDiscreteStep) {
	internal::TrackbarParams aParams = internal::makeTrackbarParams(theMin, theMax, theDiscreteStep, theSegments, theLabelFormat, theOptions);
	return trackbar<num_type>(theWidth, theValue, aParams);
}

} // namespace cvui

#endif // _CVUI_H_

// Below this line is the implementation of all functions declared above.

#ifndef _CVUI_IMPLEMENTATION_
#define _CVUI_IMPLEMENTATION_

namespace cvui
{

// This is an internal namespace with all code
// that is shared among components/functions
namespace internal
{
	bool bitsetHas(unsigned int theBitset, unsigned int theValue) {
		return (theBitset & theValue) != 0;
	}

	void error(int theId, std::string theMessage) {
		std::cout << "[CVUI] Fatal error (code " << theId << "): " << theMessage << "\n";
		cv::waitKey(100000);
		exit(-1);
	}

	void updateLayoutFlow(cvui_block_t& theBlock, cv::Size theSize) {
		int aValue;

		if (theBlock.type == ROW) {
			aValue = theSize.width + theBlock.padding;

			theBlock.anchor.x += aValue;
			theBlock.fill.width += aValue;
			theBlock.fill.height = std::max(theSize.height, theBlock.fill.height);

		}
		else if (theBlock.type == COLUMN) {
			aValue = theSize.height + theBlock.padding;

			theBlock.anchor.y += aValue;
			theBlock.fill.height += aValue;
			theBlock.fill.width = std::max(theSize.width, theBlock.fill.width);
		}
	}

	bool blockStackEmpty() {
		return gStackCount == -1;
	}

	cvui_block_t& topBlock() {
		if (gStackCount < 0) {
			error(3, "You are using a function that should be enclosed by begin*() and end*(), but you probably forgot to call begin*().");
		}

		return gStack[gStackCount];
	}

	cvui_block_t& pushBlock() {
		return gStack[++gStackCount];
	}

	cvui_block_t& popBlock() {
		// Check if there is anything to be popped out from the stack.
		if (gStackCount < 0) {
			error(1, "Mismatch in the number of begin*()/end*() calls. You are calling one more than the other.");
		}

		return gStack[gStackCount--];
	}

	void begin(int theType, cv::Mat &theWhere, int theX, int theY, int theWidth, int theHeight, int thePadding) {
		cvui_block_t& aBlock = internal::pushBlock();

		aBlock.where = theWhere;

		aBlock.rect.x = theX;
		aBlock.rect.y = theY;
		aBlock.rect.width = theWidth;
		aBlock.rect.height = theHeight;

		aBlock.fill = aBlock.rect;
		aBlock.fill.width = 0;
		aBlock.fill.height = 0;

		aBlock.anchor.x = theX;
		aBlock.anchor.y = theY;

		aBlock.padding = thePadding;
		aBlock.type = theType;
	}

	void end(int theType) {
		cvui_block_t& aBlock = popBlock();

		if (aBlock.type != theType) {
			error(4, "Calling wrong type of end*(). E.g. endColumn() instead of endRow(). Check if your begin*() calls are matched with their appropriate end*() calls.");
		}

		// If we still have blocks in the stack, we must update
		// the current top with the dimensions that were filled by
		// the newly popped block.

		if (!blockStackEmpty()) {
			cvui_block_t& aTop = topBlock();
			cv::Size aSize;

			// If the block has rect.width < 0 or rect.heigth < 0, it means the
			// user don't want to calculate the block's width/height. It's up to
			// us do to the math. In that case, we use the block's fill rect to find
			// out the occupied space. If the block's width/height is greater than
			// zero, then the user is very specific about the desired size. In that
			// case, we use the provided width/height, no matter what the fill rect
			// actually is.
			aSize.width = aBlock.rect.width < 0 ? aBlock.fill.width : aBlock.rect.width;
			aSize.height = aBlock.rect.height < 0 ? aBlock.fill.height : aBlock.rect.height;

			updateLayoutFlow(aTop, aSize);
		}
	}

	// Find the min and max values of a vector
	void findMinMax(std::vector<double>& theValues, double *theMin, double *theMax) {
		std::vector<double>::size_type aSize = theValues.size(), i;
		double aMin = theValues[0], aMax = theValues[0];

		for (i = 0; i < aSize; i++) {
			if (theValues[i] < aMin) {
				aMin = theValues[i];
			}

			if (theValues[i] > aMax) {
				aMax = theValues[i];
			}
		}

		*theMin = aMin;
		*theMax = aMax;
	}

	cvui_label_t createLabel(const std::string &theLabel) {
		cvui_label_t aLabel;
		std::stringstream aBefore, aAfter;

		aLabel.hasShortcut = false;
		aLabel.shortcut = 0;
		aLabel.textBeforeShortcut = "";
		aLabel.textAfterShortcut = "";

		for (size_t i = 0; i < theLabel.size(); i++) {
			char c = theLabel[i];
			if ((c == '&') && (i < theLabel.size() - 1)) {
				aLabel.hasShortcut = true;
				aLabel.shortcut = theLabel[i + 1];
				++i;
			}
			else if (!aLabel.hasShortcut) {
				aBefore << c;
			}
			else {
				aAfter << c;
			}
		}

		aLabel.textBeforeShortcut = aBefore.str();
		aLabel.textAfterShortcut = aAfter.str();

		return aLabel;
	}

	cv::Scalar hexToScalar(unsigned int theColor) {
		int aAlpha = (theColor >> 24) & 0xff;
		int aRed = (theColor >> 16) & 0xff;
		int aGreen = (theColor >> 8) & 0xff;
		int aBlue = theColor & 0xff;

		return cv::Scalar(aBlue, aGreen, aRed, aAlpha);
	}

	void resetRenderingBuffer(cvui_block_t& theScreen) {
		theScreen.rect.x = 0;
		theScreen.rect.y = 0;
		theScreen.rect.width = 0;
		theScreen.rect.height = 0;

		theScreen.fill = theScreen.rect;
		theScreen.fill.width = 0;
		theScreen.fill.height = 0;

		theScreen.anchor.x = 0;
		theScreen.anchor.y = 0;

		theScreen.padding = 0;
	}


	inline double clamp01(double value)
	{
		value = value > 1. ? 1. : value;
		value = value < 0. ? 0. : value;
		return value;
	}

	inline void trackbarForceValuesAsMultiplesOfSmallStep(const TrackbarParams & theParams, long double *theValue)
	{
		if (bitsetHas(theParams.options, TRACKBAR_DISCRETE) && theParams.step != 0.) {
			long double k = (*theValue - theParams.min) / theParams.step;
			k = (long double)cvRound((double)k);
			*theValue = theParams.min + theParams.step * k;
		}
	}

	inline long double trackbarXPixelToValue(const TrackbarParams & theParams, cv::Rect & theBounding, int thePixelX)
	{
		long double ratio = (thePixelX - (long double)(theBounding.x + gTrackbarMarginX)) / (long double)(theBounding.width - 2 * gTrackbarMarginX);
		ratio = clamp01(ratio);
		long double value = theParams.min + ratio * (theParams.max - theParams.min);
		return value;
	}

	inline int trackbarValueToXPixel(const TrackbarParams & theParams, cv::Rect & theBounding, long double theValue)
	{
		long double aRatio = (theValue - theParams.min) / (theParams.max - theParams.min);
		aRatio = clamp01(aRatio);
		long double thePixelsX = (long double)theBounding.x + gTrackbarMarginX + aRatio * (long double)(theBounding.width - 2 * gTrackbarMarginX);
		return (int)thePixelsX;
	}

	int iarea(int theX, int theY, int theWidth, int theHeight) {
		// By default, return that the mouse is out of the interaction area.
		int aRet = cvui::OUT;

		// Check if the mouse is over the interaction area.
		bool aMouseIsOver = cv::Rect(theX, theY, theWidth, theHeight).contains(internal::gMouse);

		if (aMouseIsOver) {
			if (internal::gMousePressed) {
				aRet = cvui::DOWN;
			}
			else {
				aRet = cvui::OVER;
			}
		}

		// Tell if the button was clicked or not
		if (aMouseIsOver && internal::gMouseJustReleased) {
			aRet = cvui::CLICK;
		}

		return aRet;
	}

	bool button(cvui_block_t& theBlock, int theX, int theY, int theWidth, int theHeight, const cv::String& theLabel, bool theUpdateLayout) {
		// Calculate the space that the label will fill
		cv::Size aTextSize = getTextSize(theLabel, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, nullptr);

		// Make the button bit enough to house the label
		cv::Rect aRect(theX, theY, theWidth, theHeight);

		// Check the state of the button (idle, pressed, etc.)
		bool aMouseIsOver = aRect.contains(internal::gMouse);

		if (aMouseIsOver) {
			if (internal::gMousePressed) {
				render::button(theBlock, cvui::DOWN, aRect, theLabel);
				render::buttonLabel(theBlock, cvui::DOWN, aRect, theLabel, aTextSize);
			}
			else {
				render::button(theBlock, cvui::OVER, aRect, theLabel);
				render::buttonLabel(theBlock, cvui::OVER, aRect, theLabel, aTextSize);
			}
		}
		else {
			render::button(theBlock, cvui::OUT, aRect, theLabel);
			render::buttonLabel(theBlock, cvui::OUT, aRect, theLabel, aTextSize);
		}

		// Update the layout flow according to button size
		// if we were told to update.
		if (theUpdateLayout) {
			cv::Size aSize(theWidth, theHeight);
			updateLayoutFlow(theBlock, aSize);
		}

		bool wasShortcutPressed = false;

		//Handle keyboard shortcuts
		if (internal::gLastKeyPressed != -1) {
			// TODO: replace with something like strpos(). I think it has better performance.
			auto aLabel = internal::createLabel(theLabel);
			if (aLabel.hasShortcut && (tolower(aLabel.shortcut) == tolower((char)internal::gLastKeyPressed))) {
				wasShortcutPressed = true;
			}
		}

		// Tell if the button was clicked or not
		return (aMouseIsOver && internal::gMouseJustReleased) || wasShortcutPressed;
	}

	bool button(cvui_block_t& theBlock, int theX, int theY, const cv::String& theLabel) {
		// Calculate the space that the label will fill
		cv::Size aTextSize = getTextSize(theLabel, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, nullptr);

		// Create a button based on the size of the text
		return internal::button(theBlock, theX, theY, aTextSize.width + 30, aTextSize.height + 18, theLabel, true);
	}

	bool button(cvui_block_t& theBlock, int theX, int theY, cv::Mat& theIdle, cv::Mat& theOver, cv::Mat& theDown, bool theUpdateLayout) {
		cv::Rect aRect(theX, theY, theIdle.cols, theIdle.rows);
		int aStatus = cvui::iarea(theX, theY, aRect.width, aRect.height);

		switch (aStatus) {
		case cvui::OUT: render::image(theBlock, aRect, theIdle); break;
		case cvui::OVER: render::image(theBlock, aRect, theOver); break;
		case cvui::DOWN: render::image(theBlock, aRect, theDown); break;
		}

		// Update the layout flow according to button size
		// if we were told to update.
		if (theUpdateLayout) {
			cv::Size aSize(aRect.width, aRect.height);
			updateLayoutFlow(theBlock, aSize);
		}

		// Return true if the button was clicked
		return aStatus == cvui::CLICK;
	}

	void image(cvui_block_t& theBlock, int theX, int theY, cv::Mat& theImage) {
		cv::Rect aRect(theX, theY, theImage.cols, theImage.rows);

		// TODO: check for render outside the frame area
		render::image(theBlock, aRect, theImage);

		// Update the layout flow according to image size
		cv::Size aSize(theImage.cols, theImage.rows);
		updateLayoutFlow(theBlock, aSize);
	}

	bool checkbox(cvui_block_t& theBlock, int theX, int theY, const cv::String& theLabel, bool *theState, unsigned int theColor) {
		cv::Rect aRect(theX, theY, 15, 15);
		cv::Size aTextSize = getTextSize(theLabel, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, nullptr);
		cv::Rect aHitArea(theX, theY, aRect.width + aTextSize.width + 6, aRect.height);
		bool aMouseIsOver = aHitArea.contains(internal::gMouse);

		if (aMouseIsOver) {
			render::checkbox(theBlock, cvui::OVER, aRect);

			if (internal::gMouseJustReleased) {
				*theState = !(*theState);
			}
		}
		else {
			render::checkbox(theBlock, cvui::OUT, aRect);
		}

		render::checkboxLabel(theBlock, aRect, theLabel, aTextSize, theColor);

		if (*theState) {
			render::checkboxCheck(theBlock, aRect);
		}

		// Update the layout flow
		cv::Size aSize(aHitArea.width, aHitArea.height);
		updateLayoutFlow(theBlock, aSize);

		return *theState;
	}

	void text(cvui_block_t& theBlock, int theX, int theY, const cv::String& theText, double theFontScale, unsigned int theColor, bool theUpdateLayout) {
		cv::Size aTextSize = cv::getTextSize(theText, cv::FONT_HERSHEY_SIMPLEX, theFontScale, 1, nullptr);
		cv::Point aPos(theX, theY + aTextSize.height);

		render::text(theBlock, theText, aPos, theFontScale, theColor);

		if (theUpdateLayout) {
			// Add an extra pixel to the height to overcome OpenCV font size problems.
			aTextSize.height += 1;

			updateLayoutFlow(theBlock, aTextSize);
		}
	}

	int counter(cvui_block_t& theBlock, int theX, int theY, int *theValue, int theStep, const char *theFormat) {
		cv::Rect aContentArea(theX + 22, theY, 48, 22);

		if (internal::button(theBlock, theX, theY, 22, 22, "-", false)) {
			*theValue -= theStep;
		}

		sprintf_s(internal::gBuffer, theFormat, *theValue);
		render::counter(theBlock, aContentArea, internal::gBuffer);

		if (internal::button(theBlock, aContentArea.x + aContentArea.width, theY, 22, 22, "+", false)) {
			*theValue += theStep;
		}

		// Update the layout flow
		cv::Size aSize(22 * 2 + aContentArea.width, aContentArea.height);
		updateLayoutFlow(theBlock, aSize);

		return *theValue;
	}

	double counter(cvui_block_t& theBlock, int theX, int theY, double *theValue, double theStep, const char *theFormat) {
		cv::Rect aContentArea(theX + 22, theY, 48, 22);

		if (internal::button(theBlock, theX, theY, 22, 22, "-", false)) {
			*theValue -= theStep;
		}

		sprintf_s(internal::gBuffer, theFormat, *theValue);
		render::counter(theBlock, aContentArea, internal::gBuffer);

		if (internal::button(theBlock, aContentArea.x + aContentArea.width, theY, 22, 22, "+", false)) {
			*theValue += theStep;
		}

		// Update the layout flow
		cv::Size aSize(22 * 2 + aContentArea.width, aContentArea.height);
		updateLayoutFlow(theBlock, aSize);

		return *theValue;
	}

	bool trackbar(cvui_block_t& theBlock, int theX, int theY, int theWidth, long double *theValue, const TrackbarParams & theParams) {
		cv::Rect aContentArea(theX, theY, theWidth, 45);
		long double aValue = *theValue;
		bool aMouseIsOver = aContentArea.contains(internal::gMouse);

		render::trackbar(theBlock, aMouseIsOver ? OVER : OUT, aContentArea, *theValue, theParams);

		if (internal::gMousePressed && aMouseIsOver) {
			*theValue = internal::trackbarXPixelToValue(theParams, aContentArea, internal::gMouse.x);

			if (bitsetHas(theParams.options, TRACKBAR_DISCRETE)) {
				internal::trackbarForceValuesAsMultiplesOfSmallStep(theParams, theValue);
			}
		}

		// Update the layout flow
		cv::Size aSize = aContentArea.size();
		updateLayoutFlow(theBlock, aSize);

		return (*theValue != aValue);
	}


	void window(cvui_block_t& theBlock, int theX, int theY, int theWidth, int theHeight, const cv::String& theTitle) {
		cv::Rect aTitleBar(theX, theY, theWidth, 20);
		cv::Rect aContent(theX, theY + aTitleBar.height, theWidth, theHeight - aTitleBar.height);

		render::window(theBlock, aTitleBar, aContent, theTitle);

		// Update the layout flow
		cv::Size aSize(theWidth, theHeight);
		updateLayoutFlow(theBlock, aSize);
	}

	void rect(cvui_block_t& theBlock, int theX, int theY, int theWidth, int theHeight, unsigned int theBorderColor, unsigned int theFillingColor) {
		cv::Rect aRect(theX, theY, theWidth, theHeight);
		render::rect(theBlock, aRect, theBorderColor, theFillingColor);

		// Update the layout flow
		cv::Size aSize(aRect.width, aRect.height);
		updateLayoutFlow(theBlock, aSize);
	}

	void sparkline(cvui_block_t& theBlock, std::vector<double>& theValues, int theX, int theY, int theWidth, int theHeight, unsigned int theColor) {
		double aMin, aMax;
		cv::Rect aRect(theX, theY, theWidth, theHeight);

		internal::findMinMax(theValues, &aMin, &aMax);
		render::sparkline(theBlock, theValues, aRect, aMin, aMax, theColor);

		// Update the layout flow
		cv::Size aSize(theWidth, theHeight);
		updateLayoutFlow(theBlock, aSize);
	}
} // namespace internal

// This is an internal namespace with all functions
// that actually render each one of the UI components
namespace render
{
	void text(cvui_block_t& theBlock, const cv::String& theText, cv::Point& thePos, double theFontScale, unsigned int theColor) {
		cv::putText(theBlock.where, theText, thePos, cv::FONT_HERSHEY_SIMPLEX, theFontScale, internal::hexToScalar(theColor), 1, CVUI_ANTIALISED);
	}

	void button(cvui_block_t& theBlock, int theState, cv::Rect& theShape, const cv::String& theLabel) {
		// Outline
		cv::rectangle(theBlock.where, theShape, cv::Scalar(0x29, 0x29, 0x29));

		// Border
		theShape.x++; theShape.y++; theShape.width -= 2; theShape.height -= 2;
		cv::rectangle(theBlock.where, theShape, cv::Scalar(0x4A, 0x4A, 0x4A));

		// Inside
		theShape.x++; theShape.y++; theShape.width -= 2; theShape.height -= 2;
		cv::rectangle(theBlock.where, theShape, theState == OUT ? cv::Scalar(0x42, 0x42, 0x42) : (theState == OVER ? cv::Scalar(0x52, 0x52, 0x52) : cv::Scalar(0x32, 0x32, 0x32)), CVUI_FILLED);
	}

	int putText(cvui_block_t& theBlock, int theState, cv::Scalar aColor, const std::string& theText, const cv::Point & thePosition) {
		double aFontSize = theState == cvui::DOWN ? 0.39 : 0.4;
		cv::Size aSize;

		if (theText != "") {
			cv::putText(theBlock.where, theText, thePosition, cv::FONT_HERSHEY_SIMPLEX, aFontSize, aColor, 1, CVUI_ANTIALISED);
			aSize = cv::getTextSize(theText, cv::FONT_HERSHEY_SIMPLEX, aFontSize, 1, nullptr);
		}

		return aSize.width;
	}

	int putTextCentered(cvui_block_t& theBlock, const cv::Point & position, const std::string &text) {
		double aFontScale = 0.3;

		auto size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, aFontScale, 1, nullptr);
		cv::Point positionDecentered(position.x - size.width / 2, position.y);
		cv::putText(theBlock.where, text, positionDecentered, cv::FONT_HERSHEY_SIMPLEX, aFontScale, cv::Scalar(0xCE, 0xCE, 0xCE), 1, CVUI_ANTIALISED);

		return size.width;
	};

	void buttonLabel(cvui_block_t& theBlock, int theState, cv::Rect theRect, const cv::String& theLabel, cv::Size& theTextSize) {
		cv::Point aPos(theRect.x + theRect.width / 2 - theTextSize.width / 2, theRect.y + theRect.height / 2 + theTextSize.height / 2);
		cv::Scalar aColor = cv::Scalar(0xCE, 0xCE, 0xCE);

		auto aLabel = internal::createLabel(theLabel);

		if (!aLabel.hasShortcut) {
			putText(theBlock, theState, aColor, theLabel, aPos);
		}
		else {
			int aWidth = putText(theBlock, theState, aColor, aLabel.textBeforeShortcut, aPos);
			int aStart = aPos.x + aWidth;
			aPos.x += aWidth;

			std::string aShortcut;
			aShortcut.push_back(aLabel.shortcut);

			aWidth = putText(theBlock, theState, aColor, aShortcut, aPos);
			int aEnd = aStart + aWidth;
			aPos.x += aWidth;

			putText(theBlock, theState, aColor, aLabel.textAfterShortcut, aPos);
			cv::line(theBlock.where, cv::Point(aStart, aPos.y + 3), cv::Point(aEnd, aPos.y + 3), aColor, 1, CVUI_ANTIALISED);
		}
	}

	void image(cvui_block_t& theBlock, cv::Rect& theRect, cv::Mat& theImage) {
		theImage.copyTo(theBlock.where(theRect));
	}

	void counter(cvui_block_t& theBlock, cv::Rect& theShape, const cv::String& theValue) {
		cv::rectangle(theBlock.where, theShape, cv::Scalar(0x29, 0x29, 0x29), CVUI_FILLED); // fill
		cv::rectangle(theBlock.where, theShape, cv::Scalar(0x45, 0x45, 0x45)); // border

		cv::Size aTextSize = getTextSize(theValue, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, nullptr);

		cv::Point aPos(theShape.x + theShape.width / 2 - aTextSize.width / 2, theShape.y + aTextSize.height / 2 + theShape.height / 2);
		cv::putText(theBlock.where, theValue, aPos, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0xCE, 0xCE, 0xCE), 1, CVUI_ANTIALISED);
	}

	void trackbarHandle(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams, cv::Rect& theWorkingArea) {
		cv::Point aBarTopLeft(theWorkingArea.x, theWorkingArea.y + theWorkingArea.height / 2);
		int aBarHeight = 7;

		// Draw the rectangle representing the handle
		int aPixelX = internal::trackbarValueToXPixel(theParams, theShape, theValue);
		int aIndicatorWidth = 3;
		int aIndicatorHeight = 4;
		cv::Point aPoint1(aPixelX - aIndicatorWidth, aBarTopLeft.y - aIndicatorHeight);
		cv::Point aPoint2(aPixelX + aIndicatorWidth, aBarTopLeft.y + aBarHeight + aIndicatorHeight);
		cv::Rect aRect(aPoint1, aPoint2);

		int aFillColor = theState == OVER ? 0x525252 : 0x424242;

		rect(theBlock, aRect, 0x212121, 0x212121);
		aRect.x += 1; aRect.y += 1; aRect.width -= 2; aRect.height -= 2;
		rect(theBlock, aRect, 0x515151, aFillColor);

		bool aShowLabel = internal::bitsetHas(theParams.options, cvui::TRACKBAR_HIDE_VALUE_LABEL) == false;

		// Draw the handle label
		if (aShowLabel) {
			cv::Point aTextPos(aPixelX, aPoint2.y + 11);
			sprintf_s(internal::gBuffer, theParams.labelFormat.c_str(), static_cast<long double>(theValue));
			putTextCentered(theBlock, aTextPos, internal::gBuffer);
		}
	}

	void trackbarPath(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams, cv::Rect& theWorkingArea) {
		int aBarHeight = 7;
		cv::Point aBarTopLeft(theWorkingArea.x, theWorkingArea.y + theWorkingArea.height / 2);
		cv::Rect aRect(aBarTopLeft, cv::Size(theWorkingArea.width, aBarHeight));

		int aBorderColor = theState == OVER ? 0x4e4e4e : 0x3e3e3e;

		rect(theBlock, aRect, aBorderColor, 0x292929);
		cv::line(theBlock.where, cv::Point(aRect.x + 1, aRect.y + aBarHeight - 2), cv::Point(aRect.x + aRect.width - 2, aRect.y + aBarHeight - 2), cv::Scalar(0x0e, 0x0e, 0x0e));
	}

	void trackbarSteps(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams, cv::Rect& theWorkingArea) {
		cv::Point aBarTopLeft(theWorkingArea.x, theWorkingArea.y + theWorkingArea.height / 2);
		cv::Scalar aColor(0x51, 0x51, 0x51);

		bool aDiscrete = internal::bitsetHas(theParams.options, cvui::TRACKBAR_DISCRETE);
		long double aFixedStep = aDiscrete ? theParams.step : (theParams.max - theParams.min) / 20;

		// TODO: check min, max and step to prevent infinite loop.
		for (long double aValue = theParams.min; aValue <= theParams.max; aValue += aFixedStep) {
			int aPixelX = internal::trackbarValueToXPixel(theParams, theShape, aValue);
			cv::Point aPoint1(aPixelX, aBarTopLeft.y);
			cv::Point aPoint2(aPixelX, aBarTopLeft.y - 3);
			cv::line(theBlock.where, aPoint1, aPoint2, aColor);
		}
	}

	void trackbarSegmentLabel(cvui_block_t& theBlock, cv::Rect& theShape, const internal::TrackbarParams &theParams, long double theValue, cv::Rect& theWorkingArea, bool theShowLabel) {
		cv::Scalar aColor(0x51, 0x51, 0x51);
		cv::Point aBarTopLeft(theWorkingArea.x, theWorkingArea.y + theWorkingArea.height / 2);

		int aPixelX = internal::trackbarValueToXPixel(theParams, theShape, theValue);

		cv::Point aPoint1(aPixelX, aBarTopLeft.y);
		cv::Point aPoint2(aPixelX, aBarTopLeft.y - 8);
		cv::line(theBlock.where, aPoint1, aPoint2, aColor);

		if (theShowLabel)
		{
			sprintf_s(internal::gBuffer, theParams.labelFormat.c_str(), theValue);
			cv::Point aTextPos(aPixelX, aBarTopLeft.y - 11);
			putTextCentered(theBlock, aTextPos, internal::gBuffer);
		}
  }

	void trackbarSegments(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams, cv::Rect& theWorkingArea) {
		int aSegments = theParams.segments < 1 ? 1 : theParams.segments;
		long double aSegmentLength = (long double)(theParams.max - theParams.min) / (long double)aSegments;

		bool aHasMinMaxLabels = internal::bitsetHas(theParams.options, TRACKBAR_HIDE_MIN_MAX_LABELS) == false;

		// Render the min value label
		trackbarSegmentLabel(theBlock, theShape, theParams, theParams.min, theWorkingArea, aHasMinMaxLabels);

		//Draw large steps and labels
		bool aHasSegmentLabels = internal::bitsetHas(theParams.options, TRACKBAR_HIDE_SEGMENT_LABELS) == false;
		// TODO: check min, max and step to prevent infinite loop.
		for (long double aValue = theParams.min; aValue <= theParams.max; aValue += aSegmentLength) {
			trackbarSegmentLabel(theBlock, theShape, theParams, aValue, theWorkingArea, aHasSegmentLabels);
		}

		// Render the max value label
		trackbarSegmentLabel(theBlock, theShape, theParams, theParams.max, theWorkingArea, aHasMinMaxLabels);
	}

	void trackbar(cvui_block_t& theBlock, int theState, cv::Rect& theShape, double theValue, const internal::TrackbarParams &theParams) {
		cv::Rect aWorkingArea(theShape.x + internal::gTrackbarMarginX, theShape.y, theShape.width - 2 * internal::gTrackbarMarginX, theShape.height);

		trackbarPath(theBlock, theState, theShape, theValue, theParams, aWorkingArea);

		bool aHideAllLabels = internal::bitsetHas(theParams.options, cvui::TRACKBAR_HIDE_LABELS);
		bool aShowSteps = internal::bitsetHas(theParams.options, cvui::TRACKBAR_HIDE_STEP_SCALE) == false;

		if (aShowSteps && !aHideAllLabels) {
			trackbarSteps(theBlock, theState, theShape, theValue, theParams, aWorkingArea);
		}

		if (!aHideAllLabels) {
			trackbarSegments(theBlock, theState, theShape, theValue, theParams, aWorkingArea);
		}

		trackbarHandle(theBlock, theState, theShape, theValue, theParams, aWorkingArea);
	}

	void checkbox(cvui_block_t& theBlock, int theState, cv::Rect& theShape) {
		// Outline
		cv::rectangle(theBlock.where, theShape, theState == OUT ? cv::Scalar(0x63, 0x63, 0x63) : cv::Scalar(0x80, 0x80, 0x80));

		// Border
		theShape.x++; theShape.y++; theShape.width -= 2; theShape.height -= 2;
		cv::rectangle(theBlock.where, theShape, cv::Scalar(0x17, 0x17, 0x17));

		// Inside
		theShape.x++; theShape.y++; theShape.width -= 2; theShape.height -= 2;
		cv::rectangle(theBlock.where, theShape, cv::Scalar(0x29, 0x29, 0x29), CVUI_FILLED);
	}

	void checkboxLabel(cvui_block_t& theBlock, cv::Rect& theRect, const cv::String& theLabel, cv::Size& theTextSize, unsigned int theColor) {
		cv::Point aPos(theRect.x + theRect.width + 6, theRect.y + theTextSize.height + theRect.height / 2 - theTextSize.height / 2 - 1);
		text(theBlock, theLabel, aPos, 0.4, theColor);
	}

	void checkboxCheck(cvui_block_t& theBlock, cv::Rect& theShape) {
		theShape.x++; theShape.y++; theShape.width -= 2; theShape.height -= 2;
		cv::rectangle(theBlock.where, theShape, cv::Scalar(0xFF, 0xBF, 0x75), CVUI_FILLED);
	}

	void window(cvui_block_t& theBlock, cv::Rect& theTitleBar, cv::Rect& theContent, const cv::String& theTitle) {
		bool aTransparecy = false;
		double aAlpha = 0.3;
		cv::Mat aOverlay;

		// Render the title bar.
		// First the border
		cv::rectangle(theBlock.where, theTitleBar, cv::Scalar(0x4A, 0x4A, 0x4A));
		// then the inside
		theTitleBar.x++; theTitleBar.y++; theTitleBar.width -= 2; theTitleBar.height -= 2;
		cv::rectangle(theBlock.where, theTitleBar, cv::Scalar(0x21, 0x21, 0x21), CVUI_FILLED);

		// Render title text.
		cv::Point aPos(theTitleBar.x + 5, theTitleBar.y + 12);
		cv::putText(theBlock.where, theTitle, aPos, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0xCE, 0xCE, 0xCE), 1, CVUI_ANTIALISED);

		// Render the body.
		// First the border.
		cv::rectangle(theBlock.where, theContent, cv::Scalar(0x4A, 0x4A, 0x4A));

		// Then the filling.
		theContent.x++; theContent.y++; theContent.width -= 2; theContent.height -= 2;
		cv::rectangle(aOverlay, theContent, cv::Scalar(0x31, 0x31, 0x31), CVUI_FILLED);

		if (aTransparecy) {
			theBlock.where.copyTo(aOverlay);
			cv::rectangle(aOverlay, theContent, cv::Scalar(0x31, 0x31, 0x31), CVUI_FILLED);
			cv::addWeighted(aOverlay, aAlpha, theBlock.where, 1.0 - aAlpha, 0.0, theBlock.where);

		}
		else {
			cv::rectangle(theBlock.where, theContent, cv::Scalar(0x31, 0x31, 0x31), CVUI_FILLED);
		}
	}

	void rect(cvui_block_t& theBlock, cv::Rect& thePos, unsigned int theBorderColor, unsigned int theFillingColor) {
		cv::Scalar aBorder = internal::hexToScalar(theBorderColor);
		cv::Scalar aFilling = internal::hexToScalar(theFillingColor);

		bool aHasFilling = aFilling[3] != 0xff;

		if (aHasFilling) {
			cv::rectangle(theBlock.where, thePos, aFilling, CVUI_FILLED, CVUI_ANTIALISED);
		}

		// Render the border
		cv::rectangle(theBlock.where, thePos, aBorder, 1, CVUI_ANTIALISED);
	}

	void sparkline(cvui_block_t& theBlock, std::vector<double>& theValues, cv::Rect &theRect, double theMin, double theMax, unsigned int theColor) {
		std::vector<double>::size_type aSize = theValues.size(), i;
		double aGap, aPosX, aScale = 0, x, y;

		aScale = theMax - theMin;
		aGap = (double)theRect.width / aSize;
		aPosX = theRect.x;

		for (i = 0; i <= aSize - 2; i++) {
			x = aPosX;
			y = (theValues[i] - theMin) / aScale * -(theRect.height - 5) + theRect.y + theRect.height - 5;
			cv::Point aPoint1((int)x, (int)y);

			x = aPosX + aGap;
			y = (theValues[i + 1] - theMin) / aScale * -(theRect.height - 5) + theRect.y + theRect.height - 5;
			cv::Point aPoint2((int)x, (int)y);

			cv::line(theBlock.where, aPoint1, aPoint2, internal::hexToScalar(theColor));
			aPosX += aGap;
		}
	}
} // namespace render

void init(const cv::String& theWindowName, int theDelayWaitKey) {
	cv::setMouseCallback(theWindowName, handleMouse, NULL);
	internal::gDelayWaitKey = theDelayWaitKey;
	internal::gLastKeyPressed = -1;
	//TODO: init gScreen here?
}

int lastKeyPressed() {
	return internal::gLastKeyPressed;
}

bool button(cv::Mat& theWhere, int theX, int theY, const cv::String& theLabel) {
	internal::gScreen.where = theWhere;
	return internal::button(internal::gScreen, theX, theY, theLabel);
}

bool button(cv::Mat& theWhere, int theX, int theY, int theWidth, int theHeight, const cv::String& theLabel) {
	internal::gScreen.where = theWhere;
	return internal::button(internal::gScreen, theX, theY, theWidth, theHeight, theLabel, true);
}

bool button(cv::Mat& theWhere, int theX, int theY, cv::Mat& theIdle, cv::Mat& theOver, cv::Mat& theDown) {
	internal::gScreen.where = theWhere;
	return internal::button(internal::gScreen, theX, theY, theIdle, theOver, theDown, true);
}

void image(cv::Mat& theWhere, int theX, int theY, cv::Mat& theImage) {
	internal::gScreen.where = theWhere;
	return internal::image(internal::gScreen, theX, theY, theImage);
}

bool checkbox(cv::Mat& theWhere, int theX, int theY, const cv::String& theLabel, bool *theState, unsigned int theColor) {
	internal::gScreen.where = theWhere;
	return internal::checkbox(internal::gScreen, theX, theY, theLabel, theState, theColor);
}

void text(cv::Mat& theWhere, int theX, int theY, const cv::String& theText, double theFontScale, unsigned int theColor) {
	internal::gScreen.where = theWhere;
	internal::text(internal::gScreen, theX, theY, theText, theFontScale, theColor, true);
}

void printf(cv::Mat& theWhere, int theX, int theY, double theFontScale, unsigned int theColor, const char *theFmt, ...) {
	va_list aArgs;

	va_start(aArgs, theFmt);
	vsprintf_s(internal::gBuffer, theFmt, aArgs);
	va_end(aArgs);

	internal::gScreen.where = theWhere;
	internal::text(internal::gScreen, theX, theY, internal::gBuffer, theFontScale, theColor, true);
}

void printf(cv::Mat& theWhere, int theX, int theY, const char *theFmt, ...) {
	va_list aArgs;

	va_start(aArgs, theFmt);
	vsprintf_s(internal::gBuffer, theFmt, aArgs);
	va_end(aArgs);

	internal::gScreen.where = theWhere;
	internal::text(internal::gScreen, theX, theY, internal::gBuffer, 0.4, 0xCECECE, true);
}

int counter(cv::Mat& theWhere, int theX, int theY, int *theValue, int theStep, const char *theFormat) {
	internal::gScreen.where = theWhere;
	return internal::counter(internal::gScreen, theX, theY, theValue, theStep, theFormat);
}

double counter(cv::Mat& theWhere, int theX, int theY, double *theValue, double theStep, const char *theFormat) {
	internal::gScreen.where = theWhere;
	return internal::counter(internal::gScreen, theX, theY, theValue, theStep, theFormat);
}

void window(cv::Mat& theWhere, int theX, int theY, int theWidth, int theHeight, const cv::String& theTitle) {
	internal::gScreen.where = theWhere;
	internal::window(internal::gScreen, theX, theY, theWidth, theHeight, theTitle);
}

void rect(cv::Mat& theWhere, int theX, int theY, int theWidth, int theHeight, unsigned int theBorderColor, unsigned int theFillingColor) {
	internal::gScreen.where = theWhere;
	internal::rect(internal::gScreen, theX, theY, theWidth, theHeight, theBorderColor, theFillingColor);
}

void sparkline(cv::Mat& theWhere, std::vector<double>& theValues, int theX, int theY, int theWidth, int theHeight, unsigned int theColor) {
	internal::gScreen.where = theWhere;
	internal::sparkline(internal::gScreen, theValues, theX, theY, theWidth, theHeight, theColor);
}

int iarea(int theX, int theY, int theWidth, int theHeight) {
	return internal::iarea(theX, theY, theWidth, theHeight);
}

void beginRow(cv::Mat &theWhere, int theX, int theY, int theWidth, int theHeight, int thePadding) {
	internal::begin(ROW, theWhere, theX, theY, theWidth, theHeight, thePadding);
}

void endRow() {
	internal::end(ROW);
}

void beginColumn(cv::Mat &theWhere, int theX, int theY, int theWidth, int theHeight, int thePadding) {
	internal::begin(COLUMN, theWhere, theX, theY, theWidth, theHeight, thePadding);
}

void endColumn() {
	internal::end(COLUMN);
}

void beginRow(int theWidth, int theHeight, int thePadding) {
	cvui_block_t& aBlock = internal::topBlock();
	internal::begin(ROW, aBlock.where, aBlock.anchor.x, aBlock.anchor.y, theWidth, theHeight, thePadding);
}

void beginColumn(int theWidth, int theHeight, int thePadding) {
	cvui_block_t& aBlock = internal::topBlock();
	internal::begin(COLUMN, aBlock.where, aBlock.anchor.x, aBlock.anchor.y, theWidth, theHeight, thePadding);
}

void space(int theValue) {
	cvui_block_t& aBlock = internal::topBlock();
	cv::Size aSize(theValue, theValue);

	internal::updateLayoutFlow(aBlock, aSize);
}

bool button(const cv::String& theLabel) {
	cvui_block_t& aBlock = internal::topBlock();
	return internal::button(aBlock, aBlock.anchor.x, aBlock.anchor.y, theLabel);
}

bool button(int theWidth, int theHeight, const cv::String& theLabel) {
	cvui_block_t& aBlock = internal::topBlock();
	return internal::button(aBlock, aBlock.anchor.x, aBlock.anchor.y, theWidth, theHeight, theLabel, true);
}

bool button(cv::Mat& theIdle, cv::Mat& theOver, cv::Mat& theDown) {
	cvui_block_t& aBlock = internal::topBlock();
	return internal::button(aBlock, aBlock.anchor.x, aBlock.anchor.y, theIdle, theOver, theDown, true);
}

void image(cv::Mat& theImage) {
	cvui_block_t& aBlock = internal::topBlock();
	return internal::image(aBlock, aBlock.anchor.x, aBlock.anchor.y, theImage);
}

bool checkbox(const cv::String& theLabel, bool *theState, unsigned int theColor) {
	cvui_block_t& aBlock = internal::topBlock();
	return internal::checkbox(aBlock, aBlock.anchor.x, aBlock.anchor.y, theLabel, theState, theColor);
}

void text(const cv::String& theText, double theFontScale, unsigned int theColor) {
	cvui_block_t& aBlock = internal::topBlock();
	internal::text(aBlock, aBlock.anchor.x, aBlock.anchor.y, theText, theFontScale, theColor, true);
}

void printf(double theFontScale, unsigned int theColor, const char *theFmt, ...) {
	cvui_block_t& aBlock = internal::topBlock();
	va_list aArgs;

	va_start(aArgs, theFmt);
	vsprintf_s(internal::gBuffer, theFmt, aArgs);
	va_end(aArgs);

	internal::text(aBlock, aBlock.anchor.x, aBlock.anchor.y, internal::gBuffer, theFontScale, theColor, true);
}

void printf(const char *theFmt, ...) {
	cvui_block_t& aBlock = internal::topBlock();
	va_list aArgs;

	va_start(aArgs, theFmt);
	vsprintf_s(internal::gBuffer, theFmt, aArgs);
	va_end(aArgs);

	internal::text(aBlock, aBlock.anchor.x, aBlock.anchor.y, internal::gBuffer, 0.4, 0xCECECE, true);
}

int counter(int *theValue, int theStep, const char *theFormat) {
	cvui_block_t& aBlock = internal::topBlock();
	return internal::counter(aBlock, aBlock.anchor.x, aBlock.anchor.y, theValue, theStep, theFormat);
}

double counter(double *theValue, double theStep, const char *theFormat) {
	cvui_block_t& aBlock = internal::topBlock();
	return internal::counter(aBlock, aBlock.anchor.x, aBlock.anchor.y, theValue, theStep, theFormat);
}

void window(int theWidth, int theHeight, const cv::String& theTitle) {
	cvui_block_t& aBlock = internal::topBlock();
	internal::window(aBlock, aBlock.anchor.x, aBlock.anchor.y, theWidth, theHeight, theTitle);
}

void rect(int theWidth, int theHeight, unsigned int theBorderColor, unsigned int theFillingColor) {
	cvui_block_t& aBlock = internal::topBlock();
	internal::rect(aBlock, aBlock.anchor.x, aBlock.anchor.y, theWidth, theHeight, theBorderColor, theFillingColor);
}

void sparkline(std::vector<double>& theValues, int theWidth, int theHeight, unsigned int theColor) {
	cvui_block_t& aBlock = internal::topBlock();
	internal::sparkline(aBlock, theValues, aBlock.anchor.x, aBlock.anchor.y, theWidth, theHeight, theColor);
}

void update() {
	internal::gMouseJustReleased = false;

	internal::resetRenderingBuffer(internal::gScreen);

	// If we were told to keep track of the keyboard shortcuts, we
	// proceed to handle opencv event queue.
	if (internal::gDelayWaitKey > 0) {
		internal::gLastKeyPressed = cv::waitKey(internal::gDelayWaitKey);
	}

	if (!internal::blockStackEmpty()) {
		internal::error(2, "Calling update() before finishing all begin*()/end*() calls. Did you forget to call a begin*() or an end*()? Check if every begin*() has an appropriate end*() call before you call update().");
	}
}

void handleMouse(int theEvent, int theX, int theY, int theFlags, void* theData) {
	internal::gMouse.x = theX;
	internal::gMouse.y = theY;

	if (theEvent == cv::EVENT_LBUTTONDOWN || theEvent == cv::EVENT_RBUTTONDOWN) {
		internal::gMousePressed = true;

	}
	else if (theEvent == cv::EVENT_LBUTTONUP || theEvent == cv::EVENT_RBUTTONUP) {
		internal::gMouseJustReleased = true;
		internal::gMousePressed = false;
	}
}

} // namespace cvui
#endif //_CVUI_IMPLEMENTATION_
