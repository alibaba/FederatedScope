// Refer to https://github.com/jeanboydev/Android-LineChart/blob/master/app/src/main/java/com/jeanboy/linechart/LineChartView.java

package com.example.fsandroid;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.View;

import com.example.fsandroid.utils.Log;

import java.util.ArrayList;
import java.util.List;


public class MyLineChartView extends View {

    private int numSamples = 20;

    private Paint textPaint;

    private Paint gridPaint;
    private Paint leftWavePaint;
    private Paint rightWavePaint;
    private Paint tablePaint;

    private Path gridPath;
    private Path wavePath;
    private Path tablePath;

    private int tableColor = Color.LTGRAY;
    private int leftWaveColor = getResources().getColor(R.color.left_wave_color);
    private int rightWaveColor = getResources().getColor(R.color.right_wave_color);
    private int rightTextColor = getResources().getColor(R.color.right_text_color);
    private float lineWidthDP = 1f;

    private int textColor = Color.BLACK;
    private float textSizeDP = 10f;

    private List<Float> leftDataList = new ArrayList<>();
    private List<Float> rightDataList = new ArrayList<>();

    float mHeightPx, mWidthPx;

    float chartPaddingDP = 3f;
    float chartPaddingPx;

    float topTextPaddingPx;

    float mXMin = 0;
    float mXMax = numSamples - 1;
    private float horizontalIntervalPx;

    // Memory usage
    float mLeftMax = 0;
    float mLeftMin = 0;

    // CPU usage
    float mRightMax = 100;
    float mRightMin = 0;

    float rulerLengthDP = 3f;
    float rulerLengthPx;

    float textMerginDP = 1f;
    float textMerginPx;

    private boolean isPlayAnim = false;

    public MyLineChartView(Context context) {
        this(context, null);
    }

    public MyLineChartView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public MyLineChartView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        tablePaint = new Paint();
        tablePaint.setAntiAlias(true);
        tablePaint.setStyle(Paint.Style.STROKE);
        tablePaint.setColor(tableColor);
        tablePaint.setStrokeWidth(dip2px(lineWidthDP));

        leftWavePaint = new Paint();
        leftWavePaint.setAntiAlias(true);
        leftWavePaint.setStyle(Paint.Style.FILL);//STROKE描边FILL填充
        leftWavePaint.setColor(leftWaveColor);

        rightWavePaint = new Paint();
        rightWavePaint.setAntiAlias(true);
        rightWavePaint.setStyle(Paint.Style.FILL);
        rightWavePaint.setColor(rightWaveColor);

        textPaint = new Paint();
        textPaint.setAntiAlias(true);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setColor(textColor);
        textPaint.setTextSize(sp2px(textSizeDP));

        wavePath = new Path();
        tablePath = new Path();
        gridPath = new Path();

        chartPaddingPx = dip2px(chartPaddingDP);
        topTextPaddingPx = getTextHeightPx() * 1.5f;
        rulerLengthPx = dip2px(rulerLengthDP);
        textMerginPx = dip2px(textMerginDP);
        Log.d("MyLineChartView: Finish init");

        resetParam();
    }

    private void resetParam() {
        wavePath.reset();
        tablePath.reset();
        gridPath.reset();

        horizontalIntervalPx = getChartWidthPx() / (numSamples - 1f);
    }

    private int dip2px(float dipValue) {
        float scale = getResources().getDisplayMetrics().density;
        return (int) (dipValue * scale + 0.5f);
    }

    private int sp2px(float spValue) {
        float fontScale = getResources().getDisplayMetrics().scaledDensity;
        return (int) (spValue * fontScale + 0.5f);
    }

    public void appendPoint(float leftValue, float rightValue) {

    }

    private float getChartWidthPx() {
        return mWidthPx - 2f * chartPaddingPx;
    }

    private float getChartHeightPx() {
        return mHeightPx - chartPaddingPx - topTextPaddingPx;
    }

    private float getChartYEndPx() {
        return - getChartHeightPx();
    }

    private float getChartXEndPx() {
        return getChartWidthPx();
    }

    private float getLeftActY(float value) {
        return (value - mLeftMin) / (mLeftMax - mLeftMin) * getChartYEndPx();
    }

    private float getRightActY(float value) {
        return (value - mRightMin) / (mRightMax - mRightMin) * getChartYEndPx();
    }

    private float getActX(float value) {
        return (value - mXMin) / (mXMax - mXMax) * getChartXEndPx();
    }

    private float getTextHeightPx() {
        Paint.FontMetrics fontMetrics = textPaint.getFontMetrics();
        return fontMetrics.bottom - fontMetrics.top;
    }

    private float getTextOffsetYPx() {
        Paint.FontMetrics fontMetrics = textPaint.getFontMetrics();
        return - (fontMetrics.bottom + fontMetrics.top) / 2f;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.translate(chartPaddingPx, mHeightPx - chartPaddingPx);
        // draw table
        drawTable(canvas);

        if (leftDataList.size() > 0 || rightDataList.size() > 0) {
            drawWave(canvas);
        }

        drawYRulers(canvas);

        postInvalidateDelayed(10);
    }

    public void clearData() {
        leftDataList.clear();
    }

    public void addData(float leftValue, float rightValue) {
        if (leftDataList.size() >= numSamples) {
            leftDataList.remove(0);
        }
        leftDataList.add(leftValue);

        if (rightDataList.size() >= numSamples) {
            rightDataList.remove(0);
        }
        rightDataList.add(rightValue);

        Float newMaxValue = leftDataList.stream().reduce(Float::max).get();

        if (newMaxValue > mLeftMax) {
            mLeftMax = (int) (newMaxValue / 100) * 100 + 100;
        }

        refreshLayout();
    }

    private void refreshLayout() {
        resetParam();
        requestLayout();
        postInvalidate();
    }

    private float getActY(Float value, float mMin, float mMax) {
        return (value - mMin) / (mMax - mMin) * getChartYEndPx();
    }

    private void drawTable(Canvas canvas){
        // left Y axis
        float topY = getChartYEndPx() - topTextPaddingPx;

        tablePaint.setColor(leftWaveColor);
        tablePath.moveTo(0, topY);
        tablePath.lineTo(0,0);
        tablePath.moveTo(0, getChartYEndPx());
        tablePath.lineTo(rulerLengthPx, getChartYEndPx());
        canvas.drawPath(tablePath, tablePaint);

        tablePath.reset();
        tablePaint.setColor(tableColor);
        tablePath.moveTo(0, 0);
        tablePath.lineTo(getChartXEndPx(), 0);
        canvas.drawPath(tablePath, tablePaint);

        tablePath.reset();
        tablePaint.setColor(rightWaveColor);
        tablePath.moveTo(getChartXEndPx(), 0);
        tablePath.lineTo(getChartXEndPx(), topY);
        tablePath.moveTo(getChartXEndPx(), getChartYEndPx());
        tablePath.lineTo(getChartXEndPx() - rulerLengthPx, getChartYEndPx());
        canvas.drawPath(tablePath, tablePaint);
    }

    private void drawWave(Canvas canvas) {
        // right wave
        wavePath.reset();
        wavePath.moveTo(getChartXEndPx(), 0);
        float x = 0;
        for (int i = rightDataList.size() - 1; i >= 0; i--) {
            float y = getActY(rightDataList.get(i), mRightMin, mRightMax);
            x = getChartXEndPx() - (rightDataList.size() - 1 - i) * horizontalIntervalPx;

            wavePath.lineTo(x, y);
        }
        wavePath.lineTo(x, 0);
        wavePath.close();

        canvas.drawPath(wavePath, rightWavePaint);

        wavePath.reset();

        // left wave
        wavePath.moveTo(getChartXEndPx(), 0);
        x = 0;
        for (int i = leftDataList.size() - 1; i >= 0; i--) {
            float y = getActY(leftDataList.get(i), mLeftMin, mLeftMax);
            x = getChartXEndPx() - (leftDataList.size() - 1 - i) * horizontalIntervalPx;

            wavePath.lineTo(x, y);
        }
        wavePath.lineTo(x, 0);
        wavePath.close();

        canvas.drawPath(wavePath, leftWavePaint);


    }

    private void drawYRulers(Canvas canvas) {
        textPaint.setTextAlign(Paint.Align.RIGHT);

        // right ruler
        drawRulerYText(canvas, Paint.Align.RIGHT, rightTextColor, "CPU(%)", getChartXEndPx()-textMerginPx, getChartYEndPx()-getTextHeightPx());
        drawRulerYText(canvas, Paint.Align.RIGHT, rightTextColor, Integer.toString((int)mRightMax), getChartXEndPx()-rulerLengthPx-textMerginPx, getChartYEndPx());
        // left ruler
        drawRulerYText(canvas, Paint.Align.LEFT, leftWaveColor, "Memory(MB)", textMerginPx, getChartYEndPx()-getTextHeightPx());
        drawRulerYText(canvas, Paint.Align.LEFT, leftWaveColor, Integer.toString((int)mLeftMax), rulerLengthPx+textMerginPx, getChartYEndPx());
    }



    private void drawRulerYText(Canvas canvas, Paint.Align align, int color, String text, float x, float y) {
        textPaint.setTextAlign(align);
        textPaint.setAntiAlias(true);
        textPaint.setColor(color);
        float offsetY = getTextOffsetYPx();
        float newY = y + offsetY;
        float newX = x;
        canvas.drawText(text, newX, newY, textPaint);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        Log.d("onSizeChanged: " + w + " " + h + " " + oldw + " " + oldh);
        mWidthPx = w;
        mHeightPx = h;
    }
}