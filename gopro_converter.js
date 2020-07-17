const gpmfExtract = require('gpmf-extract');
const goproTelemetry = require(`gopro-telemetry`);
const fs = require('fs');
const { exit } = require('process');


var inputFileName = process.argv.slice(2)[0];
var outputFileName = process.argv.slice(2)[1];

const file = fs.readFileSync(inputFileName);
gpmfExtract(file)
    .then(extracted => {
        goproTelemetry(extracted, { preset: 'csv' }, telemetry => {
            mergedHeaderStrings = [];
            mergedValueStrings = [];

            ['Camera-ACCL', 'Camera-GYRO', 'Camera-GPS5'].forEach(field => {
                values = telemetry[field].split("\n");
                mergedHeaderStrings.push(values[0]);
                values.shift();

                values.forEach(function (valueString, i) {
                    if (!mergedValueStrings[i]) {
                        if (i > mergedValueStrings.length) {
                            console.log("index error", i, mergedValueStrings.length)
                        }
                        mergedValueStrings.push(valueString)
                    } else {
                        mergedValueStrings[i] += ',' + valueString;
                    }
                });
            });


            finalCsv = mergedHeaderStrings.join(',') + "\n" + mergedValueStrings.join('\n');

            fs.writeFileSync(outputFileName, finalCsv);
            console.log('Telemetry saved as CSV');

        });
    })
  .catch (error => console.error(error));