const gpmfExtract = require('gpmf-extract');
const goproTelemetry = require(`gopro-telemetry`);
const fs = require('fs');
const { exit } = require('process');
const eol = require('eol');

var inputFileName = process.argv.slice(2)[0];
var outputFileName = process.argv.slice(2)[1];

let output = '';

const readStream = fs.createReadStream(inputFileName);
readStream.on('data', function(chunk) {
  if (output.length >= 26843) {
      return;
  }
  output += chunk//eol.auto(chunk.toString('utf8'));
});

readStream.on('end', function() {
  console.log('finished reading');
  // write to file here.


    //const file = fs.readFileSync(inputFileName);
    gpmfExtract(output)
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
});