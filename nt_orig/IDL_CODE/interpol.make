.RUN cw_bgroup
.RUN cw_field
.RUN cw_pdmenu

.RUN string
.RUN swath_count
.RUN warning
.RUN calendar
.RUN parse
.RUN read_file
.RUN write_file
.RUN linear

.RUN interpol

SAVE, /ROUTINE, FILENAME='interpol.sav'
EXIT
