root_path = "/net/bs-filesvr02/export/group/hierlemann/intermediate_data/Mea1k/phornauer/SCR_rebuttal_week_4/";
sorted_dirs = dir(fullfile(root_path, "230104","*","w*"));

sorting_paths = arrayfun(@(x) fullfile(string(sorted_dirs(x).folder), sorted_dirs(x).name, "sorted"), 1:length(sorted_dirs));
chip_ids = string();
well_id = string();
recording_date = string();
N_units = string();
N_good = string();

for p = 1:length(sorting_paths)
    path_parts = strsplit(sorting_paths(p),"/");
    chip_ids(p) = path_parts(end-2);
    well = char(path_parts(end-1));
    well_id(p) = str2double(well(end)) + 1;
    recording_date(p) = path_parts(end-3);
    template_file = fullfile(sorting_paths(p),"cluster_KSLabel.tsv");
    if exist(template_file,"file")
        unit_table = readtable(template_file,"Delimiter","tab","FileType","text");
        N_units(p) = size(unit_table,1);
        N_good(p) = sum(string(unit_table.KSLabel) == "good");
    else
        N_units(p) = 0;
        N_good(p) = 0;
    end
end

%%

quant_table = table(chip_ids',well_id',recording_date',N_units',N_good','VariableNames',{'ChipID','WellID','RecordingDate','N_units','N_good'});