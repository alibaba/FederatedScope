package com.example.fsandroid;

import static com.example.fsandroid.utils.ToastUtil.showToast;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import com.example.fsandroid.configs.Config;
import com.example.fsandroid.databinding.ActivitySettingBinding;
import com.example.fsandroid.utils.ToastUtil;
import com.example.fsandroid.utils.YamlUtil;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class SettingActivity extends AppCompatActivity {

    private ActionBar actionBar;
    private ListView settingList;
    private ActivitySettingBinding binding;

    private Config config;

    private SettingAdapter settingAdapter;

    private List<RowData> mlistData = new ArrayList<>();


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivitySettingBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Action bar
        actionBar = getSupportActionBar();
        actionBar.setDisplayHomeAsUpEnabled(true);
        actionBar.setHomeButtonEnabled(true);

        settingList = binding.settingList;

        config = YamlUtil.getConfig();

        initSettingList();
    }

    private void initSettingList() {
        // Obtain all data

        Field[] fields = config.getClass().getDeclaredFields();
        for (Field field: fields) {
            // Add header
            String groupName = field.getName();
            mlistData.add(new RowData(R.layout.row_item_setting_header, groupName));

            try {
                for(Field subField: field.get(config).getClass().getDeclaredFields()) {
                    // Add items
                    String kString = subField.getName();
                    String vString = subField.get(field.get(config)).toString();
                    mlistData.add(new RowData(R.layout.row_item_setting_item, kString, vString, groupName));
                }
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }

        settingAdapter = new SettingAdapter();
        settingList.setAdapter(settingAdapter);
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                // Refresh config

                finish();
                overridePendingTransition(R.anim.activity_in_static, R.anim.activity_out_to_right);
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private class SettingAdapter extends BaseAdapter {
        private LayoutInflater mInflater;

        public SettingAdapter() {
            this.mInflater = LayoutInflater.from(getApplicationContext());
        }

        @Override
        public int getCount() {
            return mlistData.size();
        }

        @Override
        public Object getItem(int i) {
            return mlistData.get(i);
        }

        @Override
        public long getItemId(int i) {
            return i;
        }

        @Override
        public View getView(int position, View view, ViewGroup viewGroup) {
            RowData rowData = mlistData.get(position);

            ViewHolder holder;
            // TODO: Consider reuse of views
            switch (rowData.layout) {
                case R.layout.row_item_setting_item:
                    view = mInflater.inflate(R.layout.row_item_setting_item, null);
                    holder = new ViewHolder();
                    holder.settingName = (TextView) view.findViewById(R.id.setting_name);
                    holder.settingValue = (EditText) view.findViewById(R.id.setting_value);
                    view.setTag(holder);

                    holder.settingName.setText(rowData.settingName);
                    holder.settingValue.setText(rowData.settingValue);
                    holder.settingValue.addTextChangedListener(
                            new ConfigItemWatcher(
                                    position,
                                    rowData.parentGroupName,
                                    rowData.settingName, holder));

                    break;
                case R.layout.row_item_setting_header:
                    view = mInflater.inflate(R.layout.row_item_setting_header, null);
                    holder = new ViewHolder();
                    holder.groupName = (TextView) view.findViewById(R.id.group_name);
                    view.setClickable(false);
                    view.setTag(holder);

                    holder.groupName.setText(rowData.groupName.toUpperCase());

                    break;
            }
            return view;
        }
    }

    public void uptConfigValue(String parentName, String targetName, String newValue) {
        // showToast(getApplicationContext(), "Update " + parentName + "." + targetName + " with " + newValue, Toast.LENGTH_SHORT);
        try {
            // Obtain parent object
            Field field = config.getClass().getField(parentName);
            Object obj = field.get(config);
            // Set value
            Field targetField = obj.getClass().getDeclaredField(targetName);

            Object targetClass = targetField.get(obj).getClass();
            //TODO: more elegant
            if (targetClass.equals(Integer.class) || targetClass.equals(int.class)) {
                Integer finalValue = Integer.valueOf(newValue);
                targetField.set(obj, finalValue);
            } else if (targetClass.equals(Float.class) || targetClass.equals(float.class)) {
                Float finalValue = Float.valueOf(newValue);
                targetField.set(obj, finalValue);
            } else {
                // String class
                targetField.set(obj, newValue);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static class ViewHolder {
        public TextView settingName;
        public EditText settingValue;
        public TextView groupName;
    }

    private class RowData{
        private int layout;
        private String settingName;
        private String settingValue;
        private String groupName;
        private String parentGroupName;

        public RowData(int layout, String groupName) {
            this.layout = layout;
            this.groupName = groupName;
        }

        public RowData(int layout, String settingName, String settingValue, String parentGroupName) {
            this.layout = layout;
            this.settingName = settingName;
            this.settingValue = settingValue;
            this.parentGroupName = parentGroupName;
        }
    }

    private class ConfigItemWatcher implements TextWatcher {
        private String parentGroupName;
        private String settingName;
        private ViewHolder viewHolder;
        private int position;

        public ConfigItemWatcher(int position, String parentGroupName, String settingName, ViewHolder viewHolder) {
            this.parentGroupName = parentGroupName;
            this.settingName = settingName;
            this.viewHolder = viewHolder;
            this.position = position;
        }

        @Override
        public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

        }

        @Override
        public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {
        }

        @Override
        public void afterTextChanged(Editable editable) {
            // showToast(getApplicationContext(), settingName + " is changed!", Toast.LENGTH_SHORT);
            // update config
            uptConfigValue(parentGroupName, settingName, viewHolder.settingValue.getText().toString());
            // update rowDataList
            mlistData.get(position).settingValue = viewHolder.settingValue.getText().toString();
        }
    }
}
