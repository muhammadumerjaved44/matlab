function dateTime = currentDateTime()
dateTime = datetime('now','TimeZone','local','Format','eeee, MMMM d, yyyy h:mm a');
% dateTime = datestr(dateTime, 'mmmm/dd/yyyy HH:MM:SS AM');
% dateTime = ['' datestr(dateTime, 'mmmm/dd/yyyy HH:MM:SS AM') ''];

% dateTime = datetime('now','TimeZone','local');


end